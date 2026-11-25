import copy
import logging
import os
import re
from urllib.parse import quote
from rdflib import BNode, Graph, Literal, RDF, URIRef, XSD, DCTERMS
from rdflib.collection import Collection
from datetime import datetime, timezone
try:
    from typing import Self
except ImportError:
    # fallback for Self in python 3.10
    from typing import TypeVar
    Self = TypeVar("Self", bound="Cube")
from typing import Tuple, Union, Callable
import pandas as pd
from rdflib import Namespace
from pylindas.lindas.namespaces import (
    CUBE,
    DCAT,
    DCT,
    META,
    Namespaces,
    QUDT,
    RELATION,
    SCHEMA,
    SH,
    TIME,
    UNIT,
    VCARD,
    VOID,
)
from pylindas.lindas.exceptions import (
    CubeAlreadyExistsError,
    DescriptionValidationError,
    ExistenceCheckError,
    HierarchyError,
)
from pylindas.pycube.schema import validate_description as validate_description_schema


def query_lindas(query: str, environment: str):
    """Thin wrapper that lazily imports the real ``query_lindas``.

    Keeps ``pylindas.pycube.cube`` importable without the ``query`` extra and
    gives tests a module-level name to patch.
    """
    from pylindas.lindas.query import query_lindas as _query_lindas
    return _query_lindas(query, environment=environment)

from pylindas.pycube.validation import dimension_columns, validate_dataframe, UNDEFINED_TOKEN
from pylindas.pycube.hierarchy import check_hierarchy, edges_from_parent_column
from pyshacl import validate

logger = logging.getLogger('pycube')


class Cube:
    _base_uri: URIRef
    _cube_uri: URIRef
    _cube_uri_no_version: URIRef # same as _cube_uri but without the version, needed for concepts handling
    _cube_dict: dict
    _graph: Graph
    _dataframe: pd.DataFrame
    _shape_dict: dict
    _shape_URI: URIRef
    _languages = (lst := ["fr", "en", "de", "it", "la", "rm"] ) + [item.upper() for item in lst]

    
    _EXISTENCE_CHECK_MODES = ("public", "none")

    def __init__(self, dataframe: pd.DataFrame, cube_yaml: dict, environment: str, local=False,
                 existence_check: str = "public", validate_description: Union[str, bool] = "warn"):
        """
        Initialize a Cube object.

        Args:
            dataframe (pd.DataFrame): The Pandas DataFrame representing the cube data.
            cube_yaml (dict): A dictionary containing cube information.
            environment (str): The environment of the cube.
            local (bool): A flag indicating whether the cube is local.
            existence_check (str): "public" (default) asks the public endpoint whether the cube
                already exists and raises CubeAlreadyExistsError; "none" sends no request.
            validate_description ("warn", "strict", True or False): Schema validation of the cube
                description, run against the packaged schema before any triple is created.
                "warn" (default): all violations are reported in ONE ``logger.warning`` (the module
                logger 'pycube', like all other problems reported by this class; ``warnings.warn``
                is deliberately not used) as ``path: message`` entries, and the cube is built anyway.
                "strict": a single ``DescriptionValidationError`` listing every violation is raised.
                True is an alias for "strict". False skips validation and logs nothing.

        Returns:
            None

        Raises:
            ExistenceCheckError: if ``existence_check`` is not "public" or "none".
            ValueError: if ``validate_description`` is not "warn", "strict", True or False.
            DescriptionValidationError: if ``validate_description`` is "strict" (or True) and the
                description violates the schema.
        """
        if existence_check not in self._EXISTENCE_CHECK_MODES:
            raise ExistenceCheckError(
                f"Invalid value {existence_check!r} for parameter 'existence_check'; "
                f"accepted values are \"public\" and \"none\"."
            )
        # Identity checks on purpose: 1 == True and 0 == False must not be accepted as booleans.
        if validate_description is True:
            validate_description = "strict"
        if not (validate_description is False or (
                isinstance(validate_description, str) and validate_description in ("warn", "strict"))):
            raise ValueError(
                f"Invalid value {validate_description!r} for parameter 'validate_description'; "
                f"accepted values are \"warn\", \"strict\", True (alias for \"strict\") and False."
            )
        # Deep copy first, so nothing below (success or error) can touch the caller's objects.
        cube_yaml = copy.deepcopy(cube_yaml)
        self._dataframe = dataframe.copy()
        self._existence_check = existence_check
        self._validate_description = validate_description
        if validate_description:
            violations = validate_description_schema(cube_yaml)
            if violations and validate_description == "strict":
                raise DescriptionValidationError(violations)
            if violations:
                logger.warning(
                    "Cube description violates the schema (%d violation(s)); building the cube anyway: %s",
                    len(violations), "; ".join(f"{v.path}: {v.message}" for v in violations))
        # Computed once, before _setup_shape_dicts pops "dimensions" from the working dict.
        self._dimension_columns = dimension_columns(cube_yaml)
        # Independent copy: _cube_dict (below) is mutated during setup, _cube_yaml is not.
        self._cube_yaml = copy.deepcopy(cube_yaml)
        validate_dataframe(self._dataframe, self._dimension_columns, self._input_key_columns(cube_yaml))
        self._setup_cube_dict(cube_yaml=cube_yaml)
        self._cube_uri, self._cube_uri_no_version = self._setup_cube_uri(
            local=local, environment=environment, existence_check=existence_check)
        assert self._cube_uri is not None
        self._setup_shape_dicts()
        self._graph = self._setup_graph()
        self._concept_tables = {}   # concept key -> (versioned URI template, table incl. URI column)
        self._pending_roots = []    # concept-mapped hierarchy roots waiting for their concept table
        # self._graph.serialize("example/mock-cube.ttl", format="turtle")

    def dimension_columns(self) -> list:
        """Column names the description expects (dimensions, then measures), unique, in order."""
        return list(self._dimension_columns)

    @staticmethod
    def _input_key_columns(cube_yaml: dict) -> list:
        """Key columns used for the missing-value and duplicate checks (Requirement 6).

        Resolution order: dimensions flagged ``key: true``; otherwise dimensions with
        ``dimension-type: Key Dimension`` (what pylindas descriptions actually use; the
        design named only ``key: true``); otherwise all dimensions.
        """
        dims = cube_yaml.get("dimensions") or {}
        flagged = [n for n, d in dims.items() if isinstance(d, dict) and d.get("key") is True]
        if not flagged:
            flagged = [n for n, d in dims.items() if isinstance(d, dict) and d.get("dimension-type") == "Key Dimension"]
        return flagged or list(dims)

    def __str__(self) -> str:
        """
        Return a string representation of the Cube object.

        This method returns a string representation of the Cube object, including its URI and name.

        Returns:
            str: A string representation of the Cube object.
        """
        how_many_triples_query = (
            "SELECT (COUNT(*) as ?Triples)"
            "WHERE {"
            "    ?s ?p ?o."
            "}"
        )
        how_many_triples = self._graph.query(how_many_triples_query).bindings[0].get("Triples").value
        output = (f"Cube Object <{self._cube_uri}> with name '{self._cube_dict.get('Name').get('en')}'.\n\n"
                  f"{self._dataframe.head()}\n"
                  f"Number of triples in Graph: {how_many_triples}")
        return output

    def prepare_data(self) -> Self:
        """
        Prepare the cube data by constructing observation URIs and applying mappings.

        This method constructs observation URIs for each row in the dataframe and applies mappings to the dataframe.

        Returns:
            self
        """
        self._construct_obs_uri()
        self._apply_mappings()
        return self

    def write_cube(self, opendataswiss=False) -> Self:
        """
        Write the cube metadata to the graph.

        This method writes the cube metadata to the graph, including its URI, name, description, publisher, creator, contributor, contact point, version, and date information.

        Returns:
            self
        """
        self._graph.add((self._cube_uri, RDF.type, CUBE.Cube))
        self._graph.add((self._cube_uri, RDF.type, SCHEMA.Dataset))
        self._graph.add((self._cube_uri, RDF.type, DCAT.Dataset))
        self._graph.add((self._cube_uri, RDF.type, VOID.Dataset))

        names = self._cube_dict.get("Name")
        for lan, name in names.items():
            self._graph.add((self._cube_uri, SCHEMA.name, Literal(name, lang=lan)))

        descriptions = self._cube_dict.get("Description")
        for lan, desc in descriptions.items():
            self._graph.add((self._cube_uri, SCHEMA.description, Literal(desc, lang=lan)))

        publisher = self._cube_dict.get("Publisher")
        for pblshr in publisher:
            self._graph.add((self._cube_uri, SCHEMA.publisher, URIRef(pblshr.get("IRI"))))

        creator = self._cube_dict.get("Creator")
        for crtr in creator:
            self._graph.add((self._cube_uri, SCHEMA.creator, URIRef(crtr.get("IRI"))))

        contributor = self._cube_dict.get("Contributor")
        for cntrbtr in contributor:
            self._graph.add((self._cube_uri, SCHEMA.contributor, URIRef(cntrbtr.get("IRI"))))

        dcat_contact_point = self._write_dcat_contact_point(self._cube_dict.get("Contact Point"))
        self._graph.add((self._cube_uri, DCAT.contactPoint, dcat_contact_point))
        schema_contact_point = self._write_schema_contact_point(self._cube_dict.get("Contact Point"))
        self._graph.add((self._cube_uri, SCHEMA.contactPoint, schema_contact_point))

        for creator in self._cube_dict.get("Creator", []):
            iri = creator.get('IRI')
            self._graph.add((self._cube_uri, DCT.creator, URIRef(iri)))

        for theme in self._cube_dict.get("Themes", []):
            iri = theme.get('IRI')
            if not iri:
                continue
            self._graph.add((self._cube_uri, DCAT.theme, URIRef(theme['IRI'])))

        version = self._cube_dict.get("Version")
        self._graph.add((self._cube_uri, SCHEMA.version, Literal(version)))

        identifier = self._cube_dict.get("Identifier")
        self._graph.add((self._cube_uri, DCTERMS.identifier, Literal(identifier)))

        today = datetime.today().strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._graph.add(
            (self._cube_uri, SCHEMA.dateCreated, Literal(self._cube_dict.get("Date Created"), datatype=XSD.date)))
        self._graph.add((self._cube_uri, SCHEMA.datePublished, Literal(today, datatype=XSD.date)))
        # todo: serialization yields improper format for datetime (with timezone as +02:00 instead of proper UTC)
        self._graph.add((self._cube_uri, SCHEMA.dateModified, Literal(now, datatype=XSD.dateTime)))

        self._graph.add((self._cube_uri, CUBE.observationSet, self._cube_uri + "/ObservationSet"))
        self._graph.add((self._cube_uri, CUBE.observationConstraint, self._shape_URI))

        if self._cube_dict.get("Visualize"):
            self._graph.add((self._cube_uri, SCHEMA.workExample, URIRef("https://ld.admin.ch/application/visualize")))

        work_status = self._cube_dict.get("Work Status")
        if work_status == "Published":
            self._graph.add((self._cube_uri, SCHEMA.creativeWorkStatus,
                             URIRef("https://ld.admin.ch/vocabulary/CreativeWorkStatus/Published")))
            if opendataswiss:
                self._add_opendata_profile()
        elif work_status == "Draft":
            self._graph.add((self._cube_uri, SCHEMA.creativeWorkStatus,
                             URIRef("https://ld.admin.ch/vocabulary/CreativeWorkStatus/Draft")))

        if self._cube_dict.get("Accrual Periodicity"):
            accrual_periodicity_uri = self._get_accrual_periodicity(self._cube_dict.get("Accrual Periodicity"))
            self._graph.add((self._cube_uri, DCT.accrualPeriodicity, accrual_periodicity_uri))
        return self

    def get_iri(self) -> URIRef:
        return self._cube_uri

    def _setup_cube_dict(self, cube_yaml: dict) -> None:
        """
        Set up the cube dictionary with the provided YAML data.

        Args:
            cube_yaml (dict): A dictionary containing cube information.

        Returns:
            None
        """
        # Normalize the base URI by stripping any trailing slashes so that
        # URI construction can consistently add a single "/" separator. This
        # avoids malformed URIs regardless of whether the user supplied a
        # trailing slash in the "Base-URI" (see issue #76).
        base_uri = str(cube_yaml.get("Base-URI")).rstrip("/")
        self._base_uri = URIRef(base_uri)
        self._cube_dict = cube_yaml

    def _setup_cube_uri(self, local: bool, environment="TEST", existence_check: str = "public") ->  Tuple[URIRef, URIRef]:
        """
        Set up the cube URI by concatenating the base URI and the cube identifier with the version.

        Returns:
            URIRef: The constructed cube URI as a URIRef object.
        """
        cube_uri_no_version = self._base_uri + "/" + str(self._cube_dict.get("Identifier"))
        cube_uri = cube_uri_no_version + "/" + str(self._cube_dict.get("Version"))

        query = f"ASK {{ <{cube_uri}> ?p ?o}}"
        if not local and existence_check == "public":
            if query_lindas(query, environment=environment) == True:
                raise CubeAlreadyExistsError(cube_uri)

        return URIRef(cube_uri), URIRef(cube_uri_no_version)
    
    def _setup_shape_dicts(self) -> None:
        """Set up shape dictionaries by extracting key dimensions from cube dictionary.
        
            This function initializes the shape dictionary, shape URI, and key dimensions list based on the cube dictionary.
        
            Returns:
                None
        """
        self._shape_dict = self._cube_dict.pop("dimensions")
        self._shape_URI = URIRef(self._cube_uri + "/shape") 
        self._key_dimensions = [dim_name for dim_name, dim in self._shape_dict.items() if dim.get("dimension-type") == "Key Dimension"]

    def _setup_graph(self) -> Graph:
        """Set up the graph by binding namespaces and returning the graph object.
        
        Returns:
            Graph: The graph object with namespaces bound.
        """
        graph = Graph(bind_namespaces="none")
        for prefix, nmspc in Namespaces.items():
            graph.bind(prefix=prefix, namespace=nmspc)
        try:
            # Bind the namespace with a trailing slash so that predicate URIs
            # (constructed as base + "/" + path) are abbreviated with the prefix.
            graph.bind(prefix=self._cube_dict.get("Namespace"), namespace=Namespace(str(self._base_uri) + "/"))
        except KeyError:
            logger.error("no Namespace")
            pass
        return graph

    def _construct_obs_uri(self) -> None:
        """Construct observation URIs for each row in the dataframe.
        
        This function constructs observation URIs for each row in the dataframe based on the cube URI and key dimensions.
        
        Returns:
            None
        """
        def make_iri(row):
            return Cube._make_obs_iri(
                self._cube_uri,
                [row[key_dim] for key_dim in self._key_dimensions],
            )

        self._dataframe['obs-uri'] = self._dataframe.apply(
            make_iri, axis=1
        )
        # _make_obs_iri already returns URIRef values; mapping URIRef is
        # idempotent and kept to guarantee behavior-preserving index contents.
        self._dataframe['obs-uri'] = self._dataframe['obs-uri'].map(URIRef)
        self._dataframe = self._dataframe.set_index("obs-uri")

    @staticmethod
    def _is_missing(value) -> bool:
        """True for a scalar missing value (None, NaN, pd.NA, NaT)."""
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_iri_dimension(dim_dict: dict) -> bool:
        """True if the dimension's values are IRIs (datatype URI, or a mapping producing shared IRIs)."""
        mapping = dim_dict.get("mapping")
        if mapping:
            return mapping.get("value-type", "Shared") == "Shared"
        return dim_dict.get("datatype") == "URI"

    @staticmethod
    def _make_obs_iri(cube_uri, key_values: list) -> URIRef:
        """Build an observation URI from a cube URI and ordered key values.

        Percent-encodes each key value (``safe=""`` encodes everything that is
        not unreserved), joins them with ``_`` in order, and wraps the result as
        a ``URIRef``. A missing value (None / NaN / pd.NA) becomes the token
        ``undefined``; ``validate_dataframe`` rejects a key column that also holds
        the real value ``"undefined"``, so IRIs stay unique.
        """
        parts = [UNDEFINED_TOKEN if Cube._is_missing(v) else quote(str(v), safe="") for v in key_values]
        return URIRef(f"{cube_uri}/observation/{'_'.join(parts)}")

    # Function for dynamic replacement of column names, in between {} by effective column values in a row
    #   template example: http://the_cube_uri/concept/airport_type/{typeOfAirport}/{typeOfAirport2nd}
    def _replace_placeholders(self, row, template):
        result = template
        placeholders = re.findall(r'\{(.*?)\}', template)  # find each place holder inbetween {}
        for placeholder in placeholders:
            if placeholder in row:
                result = result.replace(f'{{{placeholder}}}', str(row[placeholder]))
        return result

    def _concept_iri(self, row, dim_name: str, template: str):
        """Fill a concept IRI template; None if the dimension or any referenced column is missing."""
        if self._is_missing(row[dim_name]):
            return None
        for placeholder in re.findall(r'\{(.*?)\}', template):
            if placeholder in row and self._is_missing(row[placeholder]):
                return None
        return self._replace_placeholders(row, template)

    def _apply_mappings(self) -> None:
        """Apply mappings to the dataframe based on the specified mapping type.
        
        This method iterates through the dimensions in the shape dictionary and applies mappings to the dataframe if a mapping is defined for the dimension. 
        For dimensions with 'additive' mapping type, it adds a baseline URI in front of the value. For example the entry 1999 will be replaced with 
        https://ld.admin.ch/time/year/1999. 
        For dimensions with 'replace' mapping type, it replaces values in the dataframe column based on the specified replacements.
        Values are not transformed to URIRef or Literal.
        
        Returns:
            None
        """
        for dim_name, dim_dict in self._shape_dict.items():
            if "mapping" in dim_dict:
                mapping = dim_dict.get("mapping")
                match mapping.get("type"):
                    case "additive":
                        base = mapping.get("base") + "{}"
                        self._dataframe[dim_name] = self._dataframe[dim_name].map(lambda x: x if Cube._is_missing(x) else base.format(quote(str(x))))
                    case "replace":
                        # missing values stay missing (they become cube:Undefined)
                        col = self._dataframe[dim_name]
                        self._dataframe[dim_name] = col.map(mapping.get("replacements")).where(col.notna(), None)
                    case "regex":
                        pat = re.compile(mapping.get("pattern"))
                        repl = mapping.get("replacement")
                        self._dataframe[dim_name] = self._dataframe[dim_name].map(lambda x: x if Cube._is_missing(x) else re.sub(pat, repl, x))
                    case "concept":
                        # The replacement string is a URL with fields in-between {}, as for example:
                        #   /airport_type/{typeOfAirport}/{typeOfAirport2nd}
                        repl = mapping.get("replacement-automated")
                        # If the path is relative (it starts with "/"), then happen it to the cube's URL 
                        #   It also means that the concept is generated with the cube
                        #   thus also add the hard-coded "/concept" path
                        if repl.startswith("/"):
                            # cast the URIRef to a string to avoid log warnings "does not look like a valid URI"
                            repl = str(self._cube_uri) + "/concept" + repl

                        # Perform the {} placeholder replacement with the column values, for each row
                        self._dataframe[dim_name] = self._dataframe.apply(lambda row: self._concept_iri(row, dim_name, repl), axis=1)
                    case "function":
                        func = self._load_function_via_exec(mapping.get("filepath"), mapping.get("function-name"))
                        self._dataframe[dim_name] = self._dataframe[dim_name].map(
                            lambda x: x if Cube._is_missing(x) else func(x))
                        
                value_type = mapping.get("value-type", 'Shared')
                assert value_type in ['Shared', 'Literal']
                self._dataframe[dim_name] = self._dataframe[dim_name].map(
                    lambda v: v if Cube._is_missing(v) else (URIRef(v) if value_type == "Shared" else Literal(v)))

    @staticmethod
    def _load_function_via_exec(filepath: str, function_name: str) -> Callable:
        namespace = {}
        with open(filepath, "r") as f:
            code = f.read()
        exec(code, namespace)  # Execute all code in the file in this namespace
        func = namespace.get(function_name)
        if func is None:
            raise ValueError(f"Function '{function_name}' not found in {filepath}")
        return func
    
    def _write_dcat_contact_point(self, contact_dict: dict) -> BNode | URIRef:
        """Writes a contact point to the graph.
        
            Args:
                contact_dict (dict): A dictionary containing information about the contact point.
                
            Returns:
                BNode or URIRef: The created BNode or URIRef representing the contact point.
        """
        if contact_dict.get("IRI"):
            return URIRef(contact_dict.get("IRI"))
        else:
            contact_node = BNode()
            self._graph.add((contact_node, RDF.type, VCARD.Organization))
            self._graph.add((contact_node, VCARD.hasEmail, Literal(contact_dict.get("E-Mail"), datatype=XSD.string)))
            self._graph.add((contact_node, VCARD.fn, Literal(contact_dict.get("Name"), datatype=XSD.string)))
            return contact_node

    def _write_schema_contact_point(self, contact_dict: dict) -> BNode | URIRef:
        """Writes a contact point to the graph.

            Args:
                contact_dict (dict): A dictionary containing information about the contact point.

            Returns:
                BNode or URIRef: The created BNode or URIRef representing the contact point.
        """
        if contact_dict.get("IRI"):
            return URIRef(contact_dict.get("IRI"))
        else:
            contact_node = BNode()
            self._graph.add((contact_node, RDF.type, SCHEMA.ContactPoint))
            self._graph.add((contact_node, SCHEMA.email, Literal(contact_dict.get("E-Mail"), datatype=XSD.string)))
            self._graph.add((contact_node, SCHEMA.name, Literal(contact_dict.get("Name"), datatype=XSD.string)))
            return contact_node

    @staticmethod
    def _get_accrual_periodicity(periodicity: str) -> URIRef:
        """Get the URIRef for the given accrual periodicity.
        
        Args:
            periodicity (str): The periodicity of the accrual.
        
        Returns:
            URIRef: The URIRef corresponding to the accrual periodicity.
        """
        base_uri = URIRef("http://publications.europe.eu/resource/authority/frequency/")
        match periodicity:
            case "daily":
                return URIRef(base_uri + "DAILY")
            case "weekly":
                return URIRef(base_uri + "WEEKLY")
            case "monthly":
                return URIRef(base_uri + "MONTHLY")
            case "yearly": 
                return URIRef(base_uri + "ANNUAL")
            case "irregular":
                return URIRef(base_uri + "IRREG")

    def write_observations(self) -> Self:
        """Write observations to the cube.

        This function iterates over the rows in the dataframe and adds each row as an observation to the cube.

        Returns:
            Self
        """
        self._graph.add((self._cube_uri + "/ObservationSet", RDF.type, CUBE.ObservationSet))
        self._dataframe.apply(self._add_observation, axis=1)
        return self

    def serialize(self, filename: Union[str, "os.PathLike[str]", None] = None) -> Union[str, Self]:
        """Serialize the cube to Turtle, as a string or to a file.

        Without a file name the Turtle is returned as ``str`` and no file is
        touched. With a file name the Turtle is written UTF-8 encoded, replacing
        existing content, and the cube is returned so calls can be chained.

        Note: requirement 2.2 says the file variant returns ``None``. This
        method deliberately keeps returning ``self`` (existing callers chain
        ``cube.serialize("x.ttl")...``). The requirement should be aligned.

        Args:
            filename: Optional path of the file to write the cube to.

        Returns:
            The Turtle string if ``filename`` is None, otherwise ``self``.

        Raises:
            OSError: If the file cannot be written; the message names the path.
                The cube graph is left unchanged.
        """
        for _node, dim_name, root, _key in self._pending_roots:
            logger.warning(f"Dimension '{dim_name}': hierarchy root '{root}' is not written because no matching "
                           f"concept table was added with write_concept().")
        turtle = self._graph.serialize(format="turtle")
        if isinstance(turtle, bytes):
            turtle = turtle.decode("utf-8")
        if filename is None:
            return turtle
        try:
            # newline="": no newline translation, otherwise Windows turns "\n" inside
            # multi-line literals into "\r\n" and the written cube differs from the graph.
            with open(filename, "w", encoding="utf-8", newline="") as f:
                f.write(turtle)
        except OSError as e:
            raise OSError(e.errno, f"Cannot write Turtle to '{os.fspath(filename)}': {e.strerror or e}", os.fspath(filename)) from e
        return self

    def _add_observation(self, obs: pd.Series) -> None:
        """Add an observation to the cube.

        It also adds the observation URI to the observation set of the cube.
        
            Args:
                obs (pd.Series): The observation data to be added. These are the single rows from the _dataframe.
        
            Returns:
                None
        """
        # obs.name is the index of the row which was set to be the 'obs-uri'. With pandas 3 + pyarrow the index
        # comes back as plain str, so it is re-wrapped as an rdflib term.
        obs_uri = URIRef(obs.name)
        self._graph.add((self._cube_uri + "/ObservationSet", CUBE.observation, obs_uri))
        self._graph.add((obs_uri, RDF.type, CUBE.Observation))
        self._graph.add((obs_uri, CUBE.observedBy, URIRef(self._cube_dict.get("Creator")[0].get("IRI"))))

        for column in obs.keys():
            shape_column = self._get_shape_column(column)
            path = URIRef(self._base_uri + "/" + shape_column.get("path"))
            sanitized_value = self._sanitize_value(obs.get(column), shape_column.get("datatype"), shape_column.get("language"),
                                                   iri=self._is_iri_dimension(shape_column))
            self._graph.add((obs_uri, URIRef(path), sanitized_value))

    def _get_shape_column(self, column: str):
        c = self._shape_dict.get(column)
        if not c:
            logger.debug(self._shape_dict)
            raise ValueError(f'Could not find {column}')
        return c

    def write_shape(self) -> Self:
        """Write the shape of the cube to the graph.

            This function writes the shape of the cube to the graph, which is used to validate the cube as well as for
            description of dimension metadata

            Returns:
                Self
        """
        self._graph.add((self._shape_URI, RDF.type, CUBE.Constraint))
        self._graph.add((self._shape_URI, RDF.type, SH.NodeShape))

        self._graph.add((self._shape_URI, SH.closed, Literal("true", datatype=XSD.boolean)))

        observation_class_node = self._write_observation_class_shape()
        self._graph.add((self._shape_URI, SH.property, observation_class_node))

        observed_by_node = self._write_observed_by_node()
        self._graph.add((self._shape_URI, SH.property, observed_by_node))
        for dim, dim_dict in self._shape_dict.items():
            shape = self._write_dimension_shape(dim_dict, self._dataframe[dim])
            self._graph.add((self._shape_URI, SH.property, shape))
        return self
    
    def write_concept(self, concept_key: str, concept_data: pd.DataFrame):
        """Write concepts in the cube's graph
        
        Args:
            key: The concept must be defined in the cube_yaml file, as a nested key under the "Concepts" key
            concept_data: A pandas dataframe containing values related to the concept (typically from a CSV file)
        
        Returns:
            self
        """
        if not "Concepts" in self._cube_dict:
            logger.error(f"Error: call to write_concept() but the cube's information do not contain a \"Concepts\" key! The \"{concept_key}\" concepts will not be added to the graph.")
            return self
        
        concepts = self._cube_dict.get("Concepts")

        if not concept_key in concepts:
            logger.error(f"Error: call to write_concept() but the cube's information do not contain a \"{concept_key}\" nested key under the \"Concepts\" key! The \"{concept_key}\" concepts will not be added to the graph.")
            return self

        concept = concepts.get(concept_key)

        # Work on a copy: the caller's frame must not receive URI columns
        concept_data = concept_data.copy()

        # Mandatory value, will cause an exception if not found
        uri = concept.get("URI")
        nameField = concept.get("name-field")

        # Handle the URI
        # if the path is relative (it starts with "/"), then happen it to the cube's URL 
        # It also means that the concept is generated with the cube
        #   thus also add the hard-coded "/concept" path
        if uri.startswith("/"):
            # cast the URIRef to a string to avoid log warnings "does not look like a valid URI"
            uri_versioned = str(self._cube_uri) + "/concept" + uri
            # If the concept is part of the cube (URI based on the cube's URI), its URL contains the cube's version
            # Then link that version to a URI with no version as done by the Cube Creator to keep links between versions of a concept
            uri_unversioned = str(self._cube_uri_no_version) + "/concept" + uri
            concept_data["URI_UNVERSIONED"] = concept_data.apply(lambda row: self._replace_placeholders(row, uri_unversioned), axis=1)
        else:
            uri_versioned = uri


        concept_data["URI"] = concept_data.apply(lambda row: self._replace_placeholders(row, uri_versioned), axis=1)

        # Optional values
        if "multilingual" in concept:
            multilingual = concept.get("multilingual")
        else:
            multilingual = False

        if "position-field" in concept:
            positionField = concept.get("position-field")
        else:
            positionField = ""

        # Prepare the optional fields of the concept by:
        # - taking only the fields that are founc in the data
        #   remark: the key of the entry (under the other-fields entry) being the name of the field in the data
        # - handle the URI field if it starts with a "/": adding the cubeURI + "/concept/prop" + the defined relative path
        # If we wanted to take only the fields that are found in the data, we could add:
        #    for key, value in otherFields.items() if key in concept_data
        #  But this does not work now that multilingual fields are handled (the key will be suffixed with the langage tag, in the data)
        if "other-fields" in concept:
            otherFields = concept.get("other-fields")
            cubeUri =  str(self._cube_uri)
            
            otherFields_dict = {
                key: {
                    **value, 
                    "URI": cubeUri + "/concept/prop" + value["URI"] if value["URI"].startswith("/") else value["URI"]
                }
                for key, value in otherFields.items()
            }            
        else:
            otherFields_dict = {}

        # Hierarchy edges: validate everything before the first triple is added
        hierarchy_edges = self._concept_hierarchy_edges(concept_key, concept, concept_data, uri_versioned)

        # Hierarchy roots waiting for this table: resolve (and validate) before the first triple
        previous_table = self._concept_tables.get(concept_key)
        self._concept_tables[concept_key] = (uri_versioned, concept_data)
        try:
            resolved_roots = []
            for pending in self._pending_roots:
                root_iri = self._resolve_pending_root(pending)
                if root_iri is not None:
                    resolved_roots.append((pending, root_iri))
        except HierarchyError:
            if previous_table is None:
                del self._concept_tables[concept_key]
            else:
                self._concept_tables[concept_key] = previous_table
            raise

        # Add the concepts to the graph
        concept_data.apply(self._add_concept, axis=1,  args=(nameField, multilingual, positionField, otherFields_dict))

        for parent_iri, child_iri in hierarchy_edges:
            self._graph.add((parent_iri, SCHEMA.hasPart, child_iri))

        for pending, root_iri in resolved_roots:
            self._graph.add((pending[0], META.hierarchyRoot, root_iri))
            self._pending_roots.remove(pending)

        return self

    @staticmethod
    def _concept_id_field(concept_key: str, uri_template: str, concept_data: pd.DataFrame) -> str:
        """The concept ID column: the single placeholder of the URI template, else a column named ``ID``."""
        placeholders = [p for p in re.findall(r'\{(.*?)\}', uri_template) if p in concept_data.columns]
        placeholders = list(dict.fromkeys(placeholders))
        if len(placeholders) == 1:
            return placeholders[0]
        if "ID" in concept_data.columns:
            return "ID"
        raise HierarchyError(
            f"Concepts.{concept_key}: cannot determine the concept ID column for parent-field "
            f"(URI template has placeholders {placeholders}; add a column 'ID')",
            details={"placeholders": placeholders})

    def _concept_hierarchy_edges(self, concept_key: str, concept: dict, concept_data: pd.DataFrame,
                                 uri_template: str) -> list[Tuple[URIRef, URIRef]]:
        """Resolve ``parent-field`` into (parent IRI, child IRI) edges, after validating them.

        The concept ID column is the single placeholder of the URI template (or a column
        named ``ID`` if the template has several). Parent IRIs are the IRIs of the rows
        carrying that ID, so they use the same template as the children.
        No ``parent-field`` means no edges. Raises HierarchyError before anything is added.
        """
        parent_field = concept.get("parent-field")
        if not parent_field:
            return []
        if parent_field not in concept_data.columns:
            raise HierarchyError(
                f"Concepts.{concept_key}.parent-field '{parent_field}' is not a column of the concept table",
                details={"missing_column": parent_field})

        id_field = self._concept_id_field(concept_key, uri_template, concept_data)

        ids = [str(v) for v in concept_data[id_field]]
        parents = list(concept_data[parent_field])
        check_hierarchy(ids, parents)

        iri_by_id: dict[str, str] = {}
        for concept_id, iri in zip(ids, concept_data["URI"]):
            iri_by_id.setdefault(concept_id, iri)
        return [(URIRef(iri_by_id[p]), URIRef(iri_by_id[c])) for p, c in edges_from_parent_column(ids, parents)]
    
    def _add_concept(self, concept: pd.DataFrame, nameField: str, multilingual: bool, positionField:str, otherFields_dict):
        """Add a concept to the graph.
        
            Args:
                concept (pd.DataFrame): The concept data to be added (original file/csv values)
                nameField: name of the field containing the name of the concept
                multilingual: if true, a multilingual concept is created, looking for columns named 'nameField' + '_' + a pre-defined language
                positionField: name of the field that contains a position for the concept. Empty if no position to be added
                otherFields_dict: an optional dictionary value that contains other fields to add to a concept (or an empty dict if no other fields)
                    About that dictionary:
                        - the key is the name of the field in the data
                        - URI field: the URI of the property, must be prepared before hand if it is a relative path in the yaml file 
                        - multilingual:  optional, look for the fields named 'key' + '_' + a pre-defined language 

            Note: from observing concepts in Lindas, a schema:sameAs is added to a generic URL based on the cube's URL but without the version

            Returns:
                None
        """
        conceptURI = URIRef(concept.URI)
        if multilingual:
            for lang in self._languages:
                name_key = f"{nameField}_{lang}"
                if name_key in concept:
                    self._graph.add((conceptURI, URIRef(SCHEMA.name), Literal(concept.get(name_key), lang=lang.lower())))            
        else:
            self._graph.add((conceptURI, URIRef(SCHEMA.name), Literal(concept.get(nameField))))

        if positionField:
            self._graph.add((conceptURI, URIRef(SCHEMA.position), Literal(concept.get(positionField))))

        # If the concept is part of the cube (URI based on the cube's URI), its URL contains the cube's version
        # Then link that version to a URI with no version as done by the Cube Creator to keep links between versions of a concept
        if "URI_UNVERSIONED" in concept:
            self._graph.add((conceptURI, URIRef(SCHEMA.sameAs), URIRef(concept.URI_UNVERSIONED)))

        # Handling other fields/properties
        for key, value in otherFields_dict.items():
            if "multilingual" in value and value['multilingual']:
                for lang in self._languages:
                    key_lng = f"{key}_{lang}"
                    if key_lng in concept:
                        self._graph.add((conceptURI, URIRef(value['URI'] ), Literal(concept.get(key_lng), lang=lang.lower())))
            else:
                if key in concept:
                    # Note: get the datatype + language of the concept from the configuration file
                    # if multilingual, this is handeled here above
                    #   but it might happen that a field is not multilingual, but still concerns a specific language
                    #   on the other hand, if the 'language' key is not in the configuration file -> it will be a simple string with no language tag
                    sanitized_value = self._sanitize_value(concept.get(key), value.get('datatype'), value.get('language'))
                    self._graph.add((conceptURI, URIRef(value['URI'] ), sanitized_value))



        
    def check_dimension_object_property(self, dimension_name: str, property: URIRef) -> bool:
        try:
            dimension = self._get_shape_column(dimension_name) # raises an exception if dimension not found
            path = URIRef(self._base_uri + "/" + dimension.get("path"))

            # Prepare the SPARQL query
            query = f"""
            SELECT DISTINCT ?obj ?value
            WHERE {{
                ?obs <{path}> ?obj .
                FILTER NOT EXISTS {{?obj <{property}> ?value}}
            }}
            """

            # Execute the query
            results = self._graph.query(query)

            logger.info("\nChecking links to concept table")
            logger.info("--------------------------------")
            logger.info("Checking that objects of the '" + dimension_name + "' dimension link to a concept with a '" + str(property) + "' property")

            # Log the objects that have no match with that property
            allValuesFound = True
            if results: 
                allValuesFound = False
                logger.info("Result: problem with the following concept(s):")
                for row in results:
                    logger.info("- %s", row["obj"])
            else:
                logger.info("Result: no problem found")

            logger.info("--------------------------------")

            return allValuesFound
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return False

    def _write_dimension_shape(self, dim_dict: dict, values: pd.Series) -> BNode:
        """Write dimension shape based on the provided dictionary and values.
        
        Args:
            dim_dict (dict): A dictionary containing information about the dimension.
            values (pd.Series): A pandas Series containing values related to the dimension.
        
        Returns:
            BNode: The created dimension node in the graph.
        """
        dim_node = BNode()
        
        self._graph.add((dim_node, SH.minCount, Literal(1)))
        self._graph.add((dim_node, SH.maxCount, Literal(1)))

        for lan, name in dim_dict.get("name").items():
            self._graph.add((dim_node, SCHEMA.name, Literal(name, lang=lan)))
        for lan, desc in dim_dict.get("description").items():
            self._graph.add((dim_node, SCHEMA.description, Literal(desc, lang=lan)))
        
        self._graph.add((dim_node, SH.path, URIRef(self._base_uri + "/" + dim_dict.get("path"))))

        has_missing = bool(values.isna().any())
        # Node that carries the real-datatype constraints (min/max): the dimension node itself,
        # or the first sh:or branch if the dimension contains a missing value (literal dimensions).
        value_node = dim_node
        if dim_dict.get("datatype") == "URI":
            self._graph.add((dim_node, SH.nodeKind, SH.IRI))
        else:
            self._graph.add((dim_node, SH.nodeKind, SH.Literal))
            if has_missing:
                value_node = BNode()
                undefined_node = BNode()
                self._graph.add((value_node, SH.datatype, XSD[dim_dict.get("datatype")]))
                self._graph.add((undefined_node, SH.datatype, CUBE.Undefined))
                or_list = BNode()
                Collection(self._graph, or_list, [value_node, undefined_node])
                self._graph.add((dim_node, SH["or"], or_list))
            else:
                self._graph.add((dim_node, SH.datatype, XSD[dim_dict.get("datatype")]))

        match dim_dict.get("dimension-type"):
            case "Key Dimension":
                self._graph.add((dim_node, RDF.type, CUBE.KeyDimension))
                
            case "Measure Dimension":
                self._graph.add((dim_node, RDF.type, CUBE.MeasureDimension))
            
            case "Annotation":
                self._graph.add((dim_node, SH.nodeKind, SH.Literal))

            case "Standard Error":
                relation_node = BNode()
                relation_path = dim_dict.get("relates-to")
                self._graph.add((relation_node, RDF.type, RELATION.StandardError))
                self._graph.add((relation_node, META.relatesTo, URIRef(self._base_uri + "/" + relation_path)))
                self._graph.add((dim_node, META.dimensionRelation, relation_node))
                self._graph.add((dim_node, SH.nodeKind, SH.Literal))
            
            # todo: in Doc beschreiben
            case "Upper uncertainty":
                relation_node = BNode()
                relation_path = dim_dict.get("relates-to")
                self._graph.add((relation_node, RDF.type, RELATION.ConfidenceUpperBound))
                self._graph.add((relation_node, META.relatesTo, URIRef(self._base_uri + "/" + relation_path)))
                self._graph.add((dim_node, META.dimensionRelation, relation_node))
                self._graph.add((dim_node, SH.nodeKind, SH.Literal))
                self._graph.add((relation_node, DCT.type, Literal("Confidence interval")))

            case "Lower uncertainty":
                relation_node = BNode()
                relation_path = dim_dict.get("relates-to")
                self._graph.add((relation_node, RDF.type, RELATION.ConfidenceLowerBound))
                self._graph.add((relation_node, META.relatesTo, URIRef(self._base_uri + "/" + relation_path)))
                self._graph.add((dim_node, META.dimensionRelation, relation_node))
                self._graph.add((dim_node, SH.nodeKind, SH.Literal))
                self._graph.add((relation_node, DCT.type, Literal("Confidence interval")))

            case _ as unrecognized:
                logger.error(f"Dimension Type '{unrecognized}' is not recognized")
        
        match dim_dict.get("scale-type"):
            case "nominal":
                self._graph.add((dim_node, QUDT.scaleType, QUDT.NominalScale))
                if dim_dict.get("dimension-type") == "Key Dimension":
                    self._add_sh_list(dim_dict, dim_node, values)
            case "ordinal":
                self._graph.add((dim_node, QUDT.scaleType, QUDT.OrdinalScale))
                if dim_dict.get("dimension-type") == "Key Dimension":
                    self._add_sh_list(dim_dict, dim_node, values)
            case "interval":
                self._graph.add((dim_node, QUDT.scaleType, QUDT.IntervalScale))
                self._add_min_max(dim_dict, value_node, values)
            case "ratio":
                self._graph.add((dim_node, QUDT.scaleType, QUDT.RatioScale))
                self._add_min_max(dim_dict, value_node, values)
            case _ as unrecognized:
                logger.error(f"Scale Type '{unrecognized}' is not recognized")
        
        
        # unit from https://www.qudt.org/doc/DOC_VOCAB-UNITS.html

        if dim_dict.get("unit") is not None:
            self._graph.add((dim_node, QUDT.hasUnit, getattr(UNIT, dim_dict.get("unit"))))

        try:
            data_kind = dim_dict.get("data-kind")
            try: 
                match data_kind.get("type"):
                    case "temporal":
                        data_kind_node = BNode()
                        self._graph.add((data_kind_node, RDF.type, TIME.GeneralDateTimeDescription))
                        self._graph.add((data_kind_node, TIME.unitType, TIME["unit" + data_kind.get("unit").capitalize()]))
                        self._graph.add((dim_node, META.dataKind, data_kind_node))
                    case "spatial-shape":
                        data_kind_node = BNode()
                        self._graph.add((data_kind_node, RDF.type, SCHEMA.GeoShape))
                        self._graph.add((dim_node, META.dataKind, data_kind_node))
                    case "spatial-coordinates":
                        data_kind_node = BNode()
                        self._graph.add((data_kind_node, RDF.type, SCHEMA.GeoCoordinates))
                        self._graph.add((dim_node, META.dataKind, data_kind_node))
            except AttributeError:
                pass
        except (KeyError, AttributeError):
            pass
        
        if dim_dict.get("annotation"):
            for antn in dim_dict.get("annotation"):
                annotation_node = self._write_annotation(antn, datatype=dim_dict.get("datatype"))
                self._graph.add((dim_node, META.annotation, annotation_node))

        if dim_dict.get("hierarchy"):
            for hrch in dim_dict.get("hierarchy"):
                hierarchy_node = self._write_hierarchy(hrch, dim_dict)
                self._graph.add((dim_node, META.inHierarchy, hierarchy_node))

        return dim_node

    def _write_observation_class_shape(self):
        observation_class_shape = BNode()
        self._graph.add((observation_class_shape, SH.path, RDF.type))
        self._graph.add((observation_class_shape, SH.nodeKind, SH.IRI))

        list_node = BNode()
        Collection(self._graph, list_node, [CUBE.Observation])
        self._graph.add((observation_class_shape, URIRef(SH + "in"), list_node))
        return observation_class_shape

    def _write_observed_by_node(self):
        observed_by_node = BNode()
        self._graph.add((observed_by_node, SH.path, CUBE.observedBy))
        self._graph.add((observed_by_node, SH.nodeKind, SH.IRI))

        list_node = BNode()
        Collection(self._graph, list_node, [URIRef(self._cube_dict.get("Creator")[0].get("IRI"))])
        self._graph.add((observed_by_node, URIRef(SH + "in"), list_node))
        return observed_by_node

    def _write_hierarchy(self, hierarchy_dict:dict, dim_dict = None) -> BNode:
        hierarchy_node = BNode()
        self._graph.add((hierarchy_node, RDF.type, META.Hierarchy))

        root = str(hierarchy_dict.get("root"))
        if root.startswith("http"):
            self._graph.add((hierarchy_node, META.hierarchyRoot, URIRef(root)))
        elif (dim_dict or {}).get("mapping", {}).get("type") == "concept":
            # The root is a concept ID. Its IRI is the IRI of the concept row with that ID
            # (same template as the concept rows). The concept table is usually written
            # after the shape, so the root is resolved, and validated, as soon as the
            # matching table is known. No hierarchyRoot triple exists before that check.
            dim_name = next((n for n, d in self._shape_dict.items() if d is dim_dict), "?")
            pending = (hierarchy_node, dim_name, root, self._template_key(self._dim_concept_template(dim_dict)))
            resolved = self._resolve_pending_root(pending)
            if resolved is not None:
                self._graph.add((hierarchy_node, META.hierarchyRoot, resolved))
            else:
                self._pending_roots.append(pending)
        else:
            mapping_dict = dim_dict.get("mapping")
            if mapping_dict.get("type") == "additive":
                root_uri = mapping_dict.get("base") + root
                self._graph.add((hierarchy_node, META.hierarchyRoot, URIRef(root_uri)))
            elif mapping_dict.get("type") == "replace":
                root_uri = mapping_dict.get("replacements").get(root)
                self._graph.add((hierarchy_node, META.hierarchyRoot, URIRef(root_uri)))

        name = hierarchy_dict.get("name")
        self._graph.add((hierarchy_node, SCHEMA.name, Literal(name)))

        next_dict = hierarchy_dict.get("next-in-hierarchy")
        self._write_next_in_hierarchy(next_dict, parent_node=hierarchy_node)
        
        return hierarchy_node

    @staticmethod
    def _template_key(template: str) -> str:
        """Placeholder names differ between a dimension mapping and its concept table; compare shapes only."""
        return re.sub(r'\{.*?\}', '{}', template)

    def _dim_concept_template(self, dim_dict: dict) -> str:
        """Absolute IRI template of a ``concept`` mapping (same logic as ``_apply_mappings``)."""
        repl = dim_dict.get("mapping").get("replacement-automated")
        if repl.startswith("/"):
            repl = str(self._cube_uri) + "/concept" + repl
        return repl

    def _resolve_pending_root(self, pending) -> URIRef | None:
        """Return the root IRI if a matching concept table is registered, else None.

        Raises HierarchyError (naming dimension and value) if the table has no concept with the root ID.
        """
        _node, dim_name, root, key = pending
        for concept_key, (template, table) in self._concept_tables.items():
            if self._template_key(template) != key:
                continue
            id_field = self._concept_id_field(concept_key, template, table)
            for concept_id, iri in zip(table[id_field], table["URI"]):
                if str(concept_id) == root:
                    return URIRef(iri)
            raise HierarchyError(
                f"Dimension '{dim_name}': hierarchy root '{root}' is not an ID in the concept table "
                f"'{concept_key}'",
                details={"dimension": dim_name, "root": root, "concept": concept_key})
        return None

    def _write_next_in_hierarchy(self, next_dict: dict, parent_node: BNode):
        next_node = BNode()
        self._graph.add((next_node, SCHEMA.name, Literal(next_dict.get("name"))))
        
        # path from top to bottom
        if "path" in next_dict:
            self._graph.add((next_node, SH.path, URIRef(next_dict.get("path"))))

        # inverse path from bottom to top
        if "inverse-path" in next_dict:
            inverse_node = BNode()
            self._graph.add((next_node, SH.path, inverse_node))
            self._graph.add((inverse_node, SH.inversePath, URIRef(next_dict.get("inverse-path"))))
        
        # target class of the next node
        if "target-class" in next_dict:
            self._graph.add((next_node, SH.targetClass, URIRef(next_dict.get("target-class"))))

        self._graph.add((parent_node, META.nextInHierarchy, next_node))

        if next_dict.get("next-in-hierarchy"):
            self._write_next_in_hierarchy(next_dict.get("next-in-hierarchy"), next_node)

    def _write_annotation(self, annotation_dict: dict, datatype: str) -> BNode:
        annotation_node = BNode()
        for lan, name in annotation_dict.get("name").items():
            self._graph.add((annotation_node, SCHEMA.name, Literal(name, lang=lan)))

        if annotation_dict.get("context"):
            for dimension, context in annotation_dict.get("context").items():
                dimension_dict = self._shape_dict.get(dimension)

                context_node = self._write_context_node(dimension_dict, context)

                self._graph.add((annotation_node, META.annotationContext, context_node))

        match annotation_dict.get("type"):
            case "limit":
                self._graph.add((annotation_node, RDF.type, META.Limit))
                value = self._sanitize_value(annotation_dict.get("value"), datatype=datatype)
                self._graph.add((annotation_node, SCHEMA.value, value))
            case "limit-range":
                self._graph.add((annotation_node, RDF.type, META.Limit))
                min_value = self._sanitize_value(annotation_dict.get("min-value"), datatype=datatype)
                self._graph.add((annotation_node, SCHEMA.minValue, min_value))
                max_value = self._sanitize_value(annotation_dict.get("max-value"), datatype=datatype)
                self._graph.add((annotation_node, SCHEMA.maxValue, max_value))

        return annotation_node

    def _write_context_node(self, dimension_dict: dict, context: Union[dict, int, float, str]):
        dimension_path = dimension_dict.get("path")
        context_node = BNode()
        self._graph.add((context_node, SH.path, URIRef(self._base_uri + "/" + dimension_path)))
        type_of_mapping = dimension_dict.get("mapping").get("type")

        match context:
            case int() | float() | str():
                match type_of_mapping:
                    case "additive":
                        context = dimension_dict.get("mapping").get("base") + str(context)
                    case "replace":
                        context = dimension_dict.get("mapping").get("replacements").get(context)
                self._graph.add((context_node, SH.hasValue, URIRef(context)))
            case dict():
                # for now, assume that the context when given as dict is a min and max
                _min = context.get("min")
                _max = context.get("max")
                match type_of_mapping:
                    case"additive":
                        _min = dimension_dict.get("mapping").get("base") + str(_min)
                        _max = dimension_dict.get("mapping").get("base") + str(_max)
                    case "replace":
                        _min = dimension_dict.get("mapping").get("base") + str(_min)
                        _max = dimension_dict.get("mapping").get("base") + str(_max)
                self._graph.add((context_node, SH.minInclusive, URIRef(_min)))
                self._graph.add((context_node, SH.maxInclusive, URIRef(_max)))

        return context_node
    
    def _add_sh_list(self, dimension_dict: dict, dim_node: BNode, values: pd.Series):
        """Add a SHACL list of all unique values to the given dimension node.
        
            Args:
                dim_node (BNode): The dimension node to which the SHACL list will be added.
                values (pd.Series): The values to be added to the SHACL list.
        
            Returns:
                None
        """
        list_node = BNode()
        has_missing = bool(values.isna().any())
        unique_values = values.dropna().unique()
        if dimension_dict.get("datatype") == "URI":
            items = [URIRef(vl) for vl in unique_values]
            if has_missing:
                items.insert(0, URIRef(CUBE.Undefined))
        else:
            items = [Literal(vl, datatype=XSD[dimension_dict.get("datatype")]) for vl in unique_values]
            if has_missing:
                items.insert(0, Literal("", datatype=CUBE.Undefined))
        Collection(self._graph, list_node, items)
        self._graph.add((dim_node, URIRef(SH + "in"), list_node))

    def _add_min_max(self, dim_dict: dict, dim_node: BNode, values: pd.Series):
        """Add minimum and maximum values to the given dimension node.
        
            Args:
                dim_node (BNode): The dimension node to which the values will be added.
                values (pd.Series): The series of values from which minimum and maximum will be calculated.
        
            Missing values (written as cube:Undefined) are ignored; with no real value no bound is written.
        """
        values = values.dropna()
        if values.empty:
            return
        _min = values.min()
        _max = values.max()

        _datatype = dim_dict.get("datatype")
        self._graph.add((dim_node, SH.minInclusive, Literal(_min, datatype=XSD[_datatype])))
        self._graph.add((dim_node, SH.maxInclusive, Literal(_max, datatype=XSD[_datatype])))

    @staticmethod
    def _sanitize_value(value, datatype, lang=None, iri: bool | None = None) -> Literal|URIRef:
        """Sanitize the input value to ensure it is in a valid format.
        
            Args:
                value: The value to be sanitized.
                datatype: The datatype of the value, given as string from XSD namespace (e.g. "integer", "string", "gYear", ...).
                lang: The language of the value if it is a string (e.g. "de", "fr", ...).
                iri: Whether the dimension holds IRIs; decides the form of a missing value
                    (``cube:Undefined`` IRI vs ``""^^cube:Undefined``). Defaults to ``datatype == "URI"``.
        
            Returns:
                Literal or URIRef: The sanitized value in the form of a typed or language tagged Literal or URIRef.
        """
        if iri is None:
            iri = datatype == "URI"
        if pd.isna(value):
            return URIRef(CUBE.Undefined) if iri else Literal("", datatype=CUBE.Undefined)
        elif datatype == "URI":
            return URIRef(value)
        elif lang!=None:
            return Literal(value, lang=lang)
        else:
            if datatype != None:
                return Literal(value, datatype=getattr(XSD, datatype))
            else:
                return Literal(value)

    def validate(self) -> tuple[bool, str]:
        valid, validation_text = self._validate_base()

        if self._cube_dict.get("Visualize"):
            valid_visualize, visualization_text = self._validate_visualize_profile()
            validation_text += "\n" + visualization_text
            valid = valid and valid_visualize

        if self._cube_dict.get("Opendataswiss"):
            valid_opendataswiss, opendataswiss_text = self._validate_opendata_profile()
            validation_text += "\n" + opendataswiss_text
            valid = valid and valid_opendataswiss

        if valid:
            return True, "Cube is valid."
        else:
            return False, validation_text

    def _validate_base(self, serialize_results=False):
        # first step: standalone-cube-constraint
        # Remark: standalone-cube-constraint contains standalone-constraint-constraint!
        shacl_graph = Graph()
        shacl_graph.parse("https://cube.link/latest/shape/standalone-cube-constraint", format="turtle")
        valid_cube, results_graph_cube, text_cube = validate(data_graph=self._graph, shacl_graph=shacl_graph)

        # third step: self-consistency
        # to do this, add the cube:Observations as target of the cube:Constraint
        self._graph.add((self._shape_URI, SH.targetClass, CUBE.Observation))
        consistent, results_graph_consistency, text_consistency = validate(data_graph=self._graph)
        # remove the target again
        self._graph.remove((self._shape_URI, SH.targetClass, CUBE.Observation))

        if serialize_results:
            results_graph = results_graph_cube + results_graph_consistency
            results_graph.serialize("./validation_results.ttl", format="turtle")

        if valid_cube and consistent:
            return True, "Cube basics are met."
        else:
            result_text = f"{text_cube}\n{text_consistency}"
            return False, result_text

    def _validate_visualize_profile(self, serialize_results=False):
        shacl_graph = Graph()
        shacl_graph.parse("https://cube.link/latest/shape/profile-visualize", format="turtle")

        valid, results_graph, text = validate(data_graph=self._graph, shacl_graph=shacl_graph)

        if serialize_results:
            results_graph.serialize("./validation_results.ttl", format="turtle")
        return valid, text

    def _add_opendata_profile(self):
        names = self._cube_dict.get("Name")
        for lan, name in names.items():
            self._graph.add((self._cube_uri, SCHEMA.name, Literal(name, lang=lan)))

        descriptions = self._cube_dict.get("Description")
        for lan, desc in descriptions.items():
            self._graph.add((self._cube_uri, DCT.description, Literal(desc, lang=lan)))

        self._graph.add((self._cube_uri, SCHEMA.workExample, URIRef("https://ld.admin.ch/application/opendataswiss")))

    def _validate_opendata_profile(self, serialize_results=False):
        shacl_graph = Graph()
        shacl_graph.parse("https://cube.link/latest/shape/profile-opendataswiss-lindas", format="turtle")

        valid, results_graph, text = validate(data_graph=self._graph, shacl_graph=shacl_graph)

        if serialize_results:
            results_graph.serialize("./validation_results.ttl", format="turtle")
        return valid, text
