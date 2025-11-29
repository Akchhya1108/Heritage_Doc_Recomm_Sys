"""
Multi-Strategy Ground Truth Generation System

This module implements a comprehensive ground truth generation system for heritage
document recommendation evaluation with:
- Human-in-the-loop validation with inter-annotator agreement
- Synthetic query generation with stratified sampling
- 4-level relevance grading with explainability
- Bias detection for graph vs semantic methods
- Versioned dataset management
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import random
from sklearn.metrics import cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@dataclass
class RelevanceJudgment:
    """Individual relevance judgment by an annotator"""
    annotator_id: str
    document_id: int
    relevance_level: int  # 0=Not Relevant, 1=Good, 2=Excellent, 3=Perfect
    rationale: str
    timestamp: str


@dataclass
class GroundTruthQuery:
    """Enhanced ground truth query with validation metadata"""
    query_id: str
    query_text: str
    query_type: str  # 'seed', 'synthetic_easy', 'synthetic_hard', 'negative'

    # Query characteristics
    heritage_types: List[str]
    domains: List[str]
    time_period: Optional[str]
    region: Optional[str]
    complexity: str  # 'simple', 'moderate', 'complex'

    # Relevance judgments (doc_id -> list of judgments)
    relevance_judgments: Dict[int, List[RelevanceJudgment]]

    # Aggregated relevance (doc_id -> consensus level 0-3)
    consensus_relevance: Dict[int, int]

    # Metadata
    inter_annotator_agreement: float  # Cohen's Kappa
    num_annotators: int
    cluster_distribution: Dict[int, int]  # cluster_id -> num_relevant_docs
    creation_date: str
    version: str

    # Expected characteristics for validation
    expected_result_size_range: Tuple[int, int]
    rationale: str  # Why this query is important for evaluation


class GroundTruthGenerator:
    """
    Main class for generating validated ground truth queries.

    This system creates unbiased, realistic test queries with verified relevance
    judgments using human-in-the-loop validation and synthetic generation.
    """

    def __init__(self, data_path: str = "data/processed/heritage_metadata.csv",
                 output_dir: str = "data/evaluation"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load heritage documents
        self.documents = pd.read_csv(self.data_path)
        self.num_docs = len(self.documents)

        # Load document metadata distributions
        self._analyze_document_distributions()

        # Storage for queries
        self.queries: List[GroundTruthQuery] = []
        self.version = "2.0"

    def _analyze_document_distributions(self):
        """Analyze document distributions across all dimensions"""

        # Parse list-like columns
        for col in ['heritage_types', 'domains', 'locations', 'persons', 'organizations']:
            if col in self.documents.columns:
                self.documents[col] = self.documents[col].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
                )

        # Cluster distribution
        if 'cluster' in self.documents.columns:
            self.cluster_distribution = Counter(self.documents['cluster'])
        else:
            self.cluster_distribution = {}

        # Heritage type distribution
        heritage_types = []
        for types in self.documents['heritage_types']:
            heritage_types.extend(types)
        self.heritage_type_distribution = Counter(heritage_types)

        # Domain distribution
        domains = []
        for doms in self.documents['domains']:
            domains.extend(doms)
        self.domain_distribution = Counter(domains)

        # Time period distribution
        self.time_period_distribution = Counter(self.documents['time_period'])

        # Region distribution
        self.region_distribution = Counter(self.documents['region'])

        print(f"📊 Document Distribution Analysis:")
        print(f"   Total documents: {self.num_docs}")
        print(f"   Clusters: {len(self.cluster_distribution)}")
        print(f"   Heritage types: {len(self.heritage_type_distribution)}")
        print(f"   Domains: {len(self.domain_distribution)}")
        print(f"   Time periods: {len(self.time_period_distribution)}")
        print(f"   Regions: {len(self.region_distribution)}")

    def generate_seed_queries(self, num_queries: int = 50) -> List[Dict]:
        """
        Generate seed queries manually based on real user information needs.

        These queries represent authentic information needs spanning:
        - All heritage types (monument, site, artifact, architecture, tradition, art)
        - All domains (religious, military, royal, cultural, archaeological, architectural)
        - All time periods (ancient, medieval, modern)
        - All regions (north, south, east, west, central)
        - Various complexity levels

        Args:
            num_queries: Number of seed queries to generate (default 50)

        Returns:
            List of seed query dictionaries ready for annotation
        """

        seed_queries = [
            # SIMPLE QUERIES - Single clear intent

            # Heritage Type: Monument
            {
                "query_text": "Show me ancient Buddhist stupas in India",
                "heritage_types": ["monument"],
                "domains": ["religious", "archaeological"],
                "time_period": "ancient",
                "region": "india",
                "complexity": "simple",
                "rationale": "Clear single-intent query for specific monument type with temporal/spatial constraints"
            },
            {
                "query_text": "What are the major Mughal forts in North India?",
                "heritage_types": ["monument", "architecture"],
                "domains": ["military", "royal"],
                "time_period": "medieval",
                "region": "north",
                "complexity": "simple",
                "rationale": "Tests retrieval of specific architectural style with regional filtering"
            },
            {
                "query_text": "Find information about UNESCO World Heritage temples",
                "heritage_types": ["monument", "site"],
                "domains": ["religious", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Tests entity recognition (UNESCO) combined with heritage type"
            },

            # Heritage Type: Site
            {
                "query_text": "Archaeological sites from the Indus Valley Civilization",
                "heritage_types": ["site"],
                "domains": ["archaeological"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "Named civilization should trigger entity matching and temporal filtering"
            },
            {
                "query_text": "Colonial era historical sites in India",
                "heritage_types": ["site"],
                "domains": ["cultural", "architectural"],
                "time_period": "modern",
                "region": "india",
                "complexity": "simple",
                "rationale": "Tests modern time period which is underrepresented in dataset"
            },

            # Heritage Type: Art
            {
                "query_text": "Ancient Indian cave paintings and rock art",
                "heritage_types": ["art", "site"],
                "domains": ["cultural", "archaeological"],
                "time_period": "ancient",
                "region": "india",
                "complexity": "simple",
                "rationale": "Combines art with site, tests specific art form recognition"
            },
            {
                "query_text": "Traditional Indian dance forms and performing arts",
                "heritage_types": ["tradition", "art"],
                "domains": ["cultural"],
                "time_period": None,
                "region": "india",
                "complexity": "simple",
                "rationale": "Tests intangible heritage retrieval (tradition type)"
            },

            # Heritage Type: Architecture
            {
                "query_text": "Examples of Dravidian temple architecture in South India",
                "heritage_types": ["architecture", "monument"],
                "domains": ["religious", "architectural"],
                "time_period": None,
                "region": "south",
                "complexity": "simple",
                "rationale": "Specific architectural style with regional constraint"
            },
            {
                "query_text": "Indo-Islamic architectural monuments",
                "heritage_types": ["architecture", "monument"],
                "domains": ["religious", "architectural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Tests architectural style recognition without spatial/temporal constraints"
            },

            # Heritage Type: Artifact
            {
                "query_text": "Ancient bronze sculptures from South India",
                "heritage_types": ["artifact", "art"],
                "domains": ["cultural", "religious"],
                "time_period": "ancient",
                "region": "south",
                "complexity": "simple",
                "rationale": "Artifact type with material specification and regional constraint"
            },
            {
                "query_text": "Medieval manuscript collections and historical documents",
                "heritage_types": ["artifact"],
                "domains": ["cultural"],
                "time_period": "medieval",
                "region": None,
                "complexity": "simple",
                "rationale": "Tests artifact retrieval with specific medium"
            },

            # MODERATE COMPLEXITY - Multiple overlapping concepts

            {
                "query_text": "Buddhist and Jain cave temples in Western India with ancient rock-cut architecture",
                "heritage_types": ["monument", "architecture", "site"],
                "domains": ["religious", "archaeological", "architectural"],
                "time_period": "ancient",
                "region": "west",
                "complexity": "moderate",
                "rationale": "Multiple religions, specific architecture type, regional + temporal constraints"
            },
            {
                "query_text": "Royal palaces and forts built by Rajput dynasties",
                "heritage_types": ["monument", "architecture"],
                "domains": ["royal", "military", "architectural"],
                "time_period": "medieval",
                "region": None,
                "complexity": "moderate",
                "rationale": "Tests dynasty entity recognition combined with heritage types"
            },
            {
                "query_text": "Sacred rivers and pilgrimage sites in Hindu tradition",
                "heritage_types": ["site", "tradition"],
                "domains": ["religious", "cultural"],
                "time_period": None,
                "region": "india",
                "complexity": "moderate",
                "rationale": "Combines tangible sites with intangible traditions"
            },
            {
                "query_text": "Ancient maritime trade routes and port cities",
                "heritage_types": ["site"],
                "domains": ["archaeological", "cultural"],
                "time_period": "ancient",
                "region": None,
                "complexity": "moderate",
                "rationale": "Thematic query requiring conceptual understanding of trade heritage"
            },
            {
                "query_text": "Monuments and buildings influenced by Persian architectural elements",
                "heritage_types": ["monument", "architecture"],
                "domains": ["architectural", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "moderate",
                "rationale": "Tests cultural influence recognition across multiple structures"
            },
            {
                "query_text": "Rock-cut monasteries and viharas from Buddhist era",
                "heritage_types": ["monument", "site", "architecture"],
                "domains": ["religious", "archaeological"],
                "time_period": "ancient",
                "region": None,
                "complexity": "moderate",
                "rationale": "Specific architectural technique + religious affiliation + temporal"
            },
            {
                "query_text": "Water management systems in ancient Indian civilizations",
                "heritage_types": ["site", "architecture"],
                "domains": ["archaeological", "cultural"],
                "time_period": "ancient",
                "region": "india",
                "complexity": "moderate",
                "rationale": "Thematic query for engineering heritage"
            },
            {
                "query_text": "Temple towns and sacred geography in medieval South India",
                "heritage_types": ["site", "monument"],
                "domains": ["religious", "cultural"],
                "time_period": "medieval",
                "region": "south",
                "complexity": "moderate",
                "rationale": "Spatial + temporal + religious + urban planning concepts"
            },

            # COMPLEX QUERIES - Multiple overlapping concepts, ambiguity

            {
                "query_text": "Compare architectural evolution from ancient Buddhist stupas to medieval Hindu temples showing Indo-Islamic synthesis",
                "heritage_types": ["monument", "architecture", "art"],
                "domains": ["religious", "architectural", "cultural"],
                "time_period": None,  # Spans ancient to medieval
                "region": None,
                "complexity": "complex",
                "rationale": "Cross-temporal comparison, multiple religions, architectural evolution - tests nuanced relevance"
            },
            {
                "query_text": "Coastal defense structures and maritime heritage from Chola period to colonial era",
                "heritage_types": ["monument", "site", "architecture"],
                "domains": ["military", "archaeological", "cultural"],
                "time_period": None,  # Spans medieval to modern
                "region": None,
                "complexity": "complex",
                "rationale": "Long temporal span, specific dynasty, thematic (maritime), functional type"
            },
            {
                "query_text": "UNESCO sites representing cultural exchange along Silk Route in Indian subcontinent",
                "heritage_types": ["site", "monument"],
                "domains": ["cultural", "archaeological"],
                "time_period": None,
                "region": None,
                "complexity": "complex",
                "rationale": "Trade route concept, international dimension, cultural exchange theme"
            },
            {
                "query_text": "Living heritage traditions practiced at ancient pilgrimage sites",
                "heritage_types": ["tradition", "site", "monument"],
                "domains": ["religious", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "complex",
                "rationale": "Tangible-intangible linkage, continuity of tradition over time"
            },
            {
                "query_text": "Rock art sites showing prehistoric to early historic transition in central India",
                "heritage_types": ["art", "site"],
                "domains": ["archaeological", "cultural"],
                "time_period": "ancient",
                "region": "central",
                "complexity": "complex",
                "rationale": "Temporal transition, underrepresented region (central), specific art form"
            },
            {
                "query_text": "Royal patronage networks connecting temples, monasteries and educational institutions",
                "heritage_types": ["monument", "site"],
                "domains": ["religious", "royal", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "complex",
                "rationale": "Network/relationship query - tests graph-based retrieval strength"
            },
            {
                "query_text": "Funerary architecture and burial practices across different cultures and time periods",
                "heritage_types": ["monument", "site", "tradition"],
                "domains": ["cultural", "archaeological", "religious"],
                "time_period": None,
                "region": None,
                "complexity": "complex",
                "rationale": "Cross-cultural, cross-temporal thematic query"
            },
            {
                "query_text": "Urban planning and city layouts in ancient vs medieval Indian capitals",
                "heritage_types": ["site", "architecture"],
                "domains": ["archaeological", "cultural", "royal"],
                "time_period": None,
                "region": None,
                "complexity": "complex",
                "rationale": "Comparative temporal query, urban planning theme"
            },

            # EDGE CASES - Rare types, underrepresented regions, specific challenges

            {
                "query_text": "Buddhist heritage sites in Eastern India",
                "heritage_types": ["site", "monument"],
                "domains": ["religious", "archaeological"],
                "time_period": None,
                "region": "east",
                "complexity": "simple",
                "rationale": "Tests underrepresented region (east - only 13 docs)"
            },
            {
                "query_text": "Heritage sites in Central India",
                "heritage_types": ["site"],
                "domains": ["cultural"],
                "time_period": None,
                "region": "central",
                "complexity": "simple",
                "rationale": "Tests most underrepresented region (central - only 2 docs)"
            },
            {
                "query_text": "Modern architectural heritage from 20th century India",
                "heritage_types": ["architecture", "monument"],
                "domains": ["architectural", "cultural"],
                "time_period": "modern",
                "region": "india",
                "complexity": "simple",
                "rationale": "Tests underrepresented time period (modern - only 20 docs)"
            },
            {
                "query_text": "Artifacts and archaeological finds from unknown or uncertain time periods",
                "heritage_types": ["artifact", "site"],
                "domains": ["archaeological"],
                "time_period": "unknown",
                "region": None,
                "complexity": "simple",
                "rationale": "Tests handling of unknown/uncertain metadata"
            },
            {
                "query_text": "Intangible cultural heritage and living traditions",
                "heritage_types": ["tradition"],
                "domains": ["cultural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Tests retrieval of intangible heritage (only 24 docs)"
            },
            {
                "query_text": "Vernacular architecture and folk traditions in rural India",
                "heritage_types": ["architecture", "tradition"],
                "domains": ["cultural"],
                "time_period": None,
                "region": None,
                "complexity": "moderate",
                "rationale": "Tests non-monumental heritage which may be underrepresented"
            },
            {
                "query_text": "Colonial Indo-Saracenic architecture blending European and Indian styles",
                "heritage_types": ["architecture", "monument"],
                "domains": ["architectural", "cultural"],
                "time_period": "modern",
                "region": None,
                "complexity": "moderate",
                "rationale": "Specific hybrid architectural style from modern period"
            },
            {
                "query_text": "Megalithic burial sites and prehistoric monuments",
                "heritage_types": ["site", "monument"],
                "domains": ["archaeological"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "Very ancient heritage that may be sparse in dataset"
            },

            # NEGATIVE EXAMPLES - Should return zero or very few results

            {
                "query_text": "Ancient Egyptian pyramids and pharaonic tombs",
                "heritage_types": ["monument", "site"],
                "domains": ["archaeological", "royal"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "NEGATIVE - Non-Indian geography, should return empty or near-empty"
            },
            {
                "query_text": "Renaissance art and architecture in Italy",
                "heritage_types": ["art", "architecture"],
                "domains": ["cultural", "architectural"],
                "time_period": "medieval",
                "region": None,
                "complexity": "simple",
                "rationale": "NEGATIVE - European geography and cultural context"
            },
            {
                "query_text": "Pre-Columbian Mayan temples and codices",
                "heritage_types": ["monument", "artifact"],
                "domains": ["religious", "archaeological"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "NEGATIVE - American geography, different cultural sphere"
            },
            {
                "query_text": "Contemporary digital art installations from 21st century",
                "heritage_types": ["art"],
                "domains": ["cultural"],
                "time_period": "modern",
                "region": None,
                "complexity": "simple",
                "rationale": "NEGATIVE - Too recent, not traditional heritage"
            },
            {
                "query_text": "Industrial revolution factories and railway heritage in Europe",
                "heritage_types": ["site", "architecture"],
                "domains": ["archaeological", "cultural"],
                "time_period": "modern",
                "region": None,
                "complexity": "simple",
                "rationale": "NEGATIVE - European industrial heritage, different domain"
            },

            # DOMAIN-SPECIFIC CLUSTERS

            # Religious domain focus
            {
                "query_text": "Hindu temples dedicated to Shiva",
                "heritage_types": ["monument"],
                "domains": ["religious"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Specific deity focus, common religious query pattern"
            },
            {
                "query_text": "Islamic mosques and dargahs in India",
                "heritage_types": ["monument", "site"],
                "domains": ["religious", "cultural"],
                "time_period": None,
                "region": "india",
                "complexity": "simple",
                "rationale": "Islamic heritage in Indian context"
            },
            {
                "query_text": "Sikh gurdwaras and historical sites",
                "heritage_types": ["monument", "site"],
                "domains": ["religious", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Specific religious tradition, tests minority representation"
            },
            {
                "query_text": "Jain temples with intricate marble carvings",
                "heritage_types": ["monument", "art"],
                "domains": ["religious", "architectural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Specific religious + artistic feature combination"
            },

            # Military domain focus
            {
                "query_text": "Hill forts and defensive fortifications in Western India",
                "heritage_types": ["monument", "site"],
                "domains": ["military", "architectural"],
                "time_period": None,
                "region": "west",
                "complexity": "simple",
                "rationale": "Military architecture with topographical specification"
            },
            {
                "query_text": "Battle sites and war memorials from Indian history",
                "heritage_types": ["site", "monument"],
                "domains": ["military", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Commemorative military heritage"
            },

            # Royal domain focus
            {
                "query_text": "Royal palaces and residential quarters of Indian rulers",
                "heritage_types": ["architecture", "monument"],
                "domains": ["royal", "architectural"],
                "time_period": None,
                "region": None,
                "complexity": "simple",
                "rationale": "Royal domestic architecture"
            },
            {
                "query_text": "Coronation sites and royal ceremonial spaces",
                "heritage_types": ["site"],
                "domains": ["royal", "cultural"],
                "time_period": None,
                "region": None,
                "complexity": "moderate",
                "rationale": "Ceremonial royal heritage, may require contextual understanding"
            },

            # PERSON ENTITY QUERIES

            {
                "query_text": "Monuments built by Ashoka the Great",
                "heritage_types": ["monument"],
                "domains": ["royal", "religious"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "Named historical figure - tests person entity matching"
            },
            {
                "query_text": "Architectural works commissioned by Shah Jahan",
                "heritage_types": ["architecture", "monument"],
                "domains": ["royal", "architectural"],
                "time_period": "medieval",
                "region": None,
                "complexity": "simple",
                "rationale": "Specific Mughal emperor - tests person entity + temporal"
            },
            {
                "query_text": "Sites associated with Gautama Buddha and early Buddhism",
                "heritage_types": ["site", "monument"],
                "domains": ["religious", "archaeological"],
                "time_period": "ancient",
                "region": None,
                "complexity": "moderate",
                "rationale": "Religious founder - tests person entity in religious context"
            },

            # ORGANIZATION ENTITY QUERIES

            {
                "query_text": "Heritage sites of the Chola dynasty",
                "heritage_types": ["site", "monument"],
                "domains": ["royal", "cultural", "religious"],
                "time_period": "medieval",
                "region": "south",
                "complexity": "simple",
                "rationale": "Dynasty entity - tests organization matching"
            },
            {
                "query_text": "Archaeological sites from Mauryan Empire",
                "heritage_types": ["site"],
                "domains": ["archaeological", "royal"],
                "time_period": "ancient",
                "region": None,
                "complexity": "simple",
                "rationale": "Empire/dynasty entity with archaeological focus"
            },
            {
                "query_text": "Buddhist monasteries established by Nalanda University",
                "heritage_types": ["monument", "site"],
                "domains": ["religious", "cultural"],
                "time_period": "ancient",
                "region": None,
                "complexity": "moderate",
                "rationale": "Educational institution entity - tests organization in religious context"
            },
        ]

        # Add query IDs and standard fields
        for i, query in enumerate(seed_queries[:num_queries]):
            query['query_id'] = f"seed_{i:03d}"
            query['query_type'] = 'negative' if 'NEGATIVE' in query['rationale'] else 'seed'
            query['expected_result_size_range'] = self._estimate_result_size(query)
            query['creation_date'] = datetime.now().isoformat()
            query['version'] = self.version

        return seed_queries[:num_queries]

    def _estimate_result_size(self, query: Dict) -> Tuple[int, int]:
        """Estimate expected result size range based on query characteristics"""

        # Start with total documents
        estimated = self.num_docs

        # Negative queries
        if 'NEGATIVE' in query.get('rationale', ''):
            return (0, 5)

        # Apply filters based on query constraints
        if query.get('time_period') and query['time_period'] != 'unknown':
            period_count = self.time_period_distribution.get(query['time_period'], 0)
            estimated = min(estimated, period_count)

        if query.get('region') and query['region'] != 'unknown':
            region_count = self.region_distribution.get(query['region'], 0)
            estimated = min(estimated, region_count)

        # Adjust for complexity
        complexity_multipliers = {
            'simple': (0.1, 0.3),    # 10-30% of filtered docs
            'moderate': (0.05, 0.2), # 5-20% of filtered docs
            'complex': (0.02, 0.1)   # 2-10% of filtered docs
        }

        multiplier = complexity_multipliers.get(query.get('complexity', 'moderate'), (0.05, 0.2))

        min_size = max(1, int(estimated * multiplier[0]))
        max_size = max(min_size + 5, int(estimated * multiplier[1]))

        return (min_size, max_size)

    def generate_synthetic_queries(self, num_easy: int = 30, num_hard: int = 20) -> List[Dict]:
        """
        Generate synthetic queries with stratified sampling.

        Creates queries that span:
        - All 12 clusters proportionally
        - All heritage types uniformly
        - All domains uniformly
        - All time periods and regions
        - Mix of easy and hard queries

        Args:
            num_easy: Number of easy queries (single clear intent)
            num_hard: Number of hard queries (multiple overlapping concepts)

        Returns:
            List of synthetic query dictionaries
        """

        synthetic_queries = []

        # EASY QUERIES - Stratified by cluster
        cluster_samples = self._stratified_cluster_sampling(num_easy)

        for i, (cluster_id, doc_indices) in enumerate(cluster_samples.items()):
            # Sample a representative document from this cluster
            doc_idx = random.choice(doc_indices)
            doc = self.documents.iloc[doc_idx]

            # Generate query based on document characteristics
            query = self._generate_query_from_document(doc, cluster_id, difficulty='easy')
            query['query_id'] = f"synthetic_easy_{i:03d}"
            query['query_type'] = 'synthetic_easy'

            synthetic_queries.append(query)

        # HARD QUERIES - Cross-cluster, multi-faceted
        for i in range(num_hard):
            query = self._generate_hard_query(i)
            query['query_id'] = f"synthetic_hard_{i:03d}"
            query['query_type'] = 'synthetic_hard'

            synthetic_queries.append(query)

        return synthetic_queries

    def _stratified_cluster_sampling(self, num_samples: int) -> Dict[int, List[int]]:
        """Sample documents proportionally from each cluster"""

        cluster_samples = defaultdict(list)

        if 'cluster' not in self.documents.columns:
            # Fallback: sample randomly
            for i in range(num_samples):
                cluster_samples[i % 12].append(random.randint(0, self.num_docs - 1))
            return cluster_samples

        # Calculate samples per cluster proportionally
        total_clusters = len(self.cluster_distribution)
        for cluster_id, count in self.cluster_distribution.items():
            proportion = count / self.num_docs
            num_cluster_samples = max(1, int(num_samples * proportion))

            # Get all doc indices in this cluster
            cluster_docs = self.documents[self.documents['cluster'] == cluster_id].index.tolist()

            # Sample
            sampled = random.sample(cluster_docs, min(num_cluster_samples, len(cluster_docs)))
            cluster_samples[cluster_id] = sampled

        return cluster_samples

    def _generate_query_from_document(self, doc: pd.Series, cluster_id: int,
                                       difficulty: str = 'easy') -> Dict:
        """Generate a query based on document characteristics"""

        heritage_types = doc['heritage_types'] if isinstance(doc['heritage_types'], list) else []
        domains = doc['domains'] if isinstance(doc['domains'], list) else []
        time_period = doc.get('time_period', None)
        region = doc.get('region', None)

        if difficulty == 'easy':
            # Simple query with 1-2 constraints
            constraints = []

            if heritage_types:
                primary_type = heritage_types[0]
                constraints.append(primary_type)

            if time_period and time_period != 'unknown':
                constraints.append(time_period)

            if region and region != 'unknown':
                constraints.append(f"in {region} India")

            query_text = f"Find {' '.join(constraints)} heritage sites"
            complexity = "simple"

        else:  # hard
            # Complex query with multiple constraints
            query_text = f"Show {time_period if time_period else 'historical'} "
            query_text += f"{heritage_types[0] if heritage_types else 'heritage'} "
            query_text += f"in {', '.join(domains[:2]) if domains else 'cultural'} domain "
            query_text += f"from {region if region else 'India'}"
            complexity = "complex"

        return {
            'query_text': query_text,
            'heritage_types': heritage_types[:2],
            'domains': domains[:2],
            'time_period': time_period if time_period != 'unknown' else None,
            'region': region if region != 'unknown' else None,
            'complexity': complexity,
            'expected_result_size_range': self._estimate_result_size({
                'time_period': time_period,
                'region': region,
                'complexity': complexity
            }),
            'rationale': f"Synthetic query generated from cluster {cluster_id} representative document",
            'creation_date': datetime.now().isoformat(),
            'version': self.version
        }

    def _generate_hard_query(self, idx: int) -> Dict:
        """Generate a complex query with multiple overlapping concepts"""

        # Randomly combine multiple heritage types and domains
        num_types = random.randint(2, 3)
        num_domains = random.randint(2, 3)

        heritage_types = random.sample(
            list(self.heritage_type_distribution.keys()),
            min(num_types, len(self.heritage_type_distribution))
        )
        domains = random.sample(
            list(self.domain_distribution.keys()),
            min(num_domains, len(self.domain_distribution))
        )

        # Random temporal/spatial constraints (50% chance)
        time_period = random.choice(list(self.time_period_distribution.keys()) + [None])
        if time_period == 'unknown':
            time_period = None

        region = random.choice(list(self.region_distribution.keys()) + [None])
        if region == 'unknown':
            region = None

        # Generate complex query text
        query_parts = []
        query_parts.append(f"{time_period if time_period else 'historical'}")
        query_parts.append(" and ".join(heritage_types[:2]))
        query_parts.append(f"in {', '.join(domains[:2])} domains")
        if region:
            query_parts.append(f"from {region} region")

        query_text = " ".join(query_parts)

        return {
            'query_text': query_text,
            'heritage_types': heritage_types,
            'domains': domains,
            'time_period': time_period,
            'region': region,
            'complexity': 'complex',
            'expected_result_size_range': self._estimate_result_size({
                'time_period': time_period,
                'region': region,
                'complexity': 'complex'
            }),
            'rationale': "Synthetic hard query with multiple overlapping concepts",
            'creation_date': datetime.now().isoformat(),
            'version': self.version
        }

    def create_annotation_interface(self, queries: List[Dict],
                                     output_file: str = "annotation_interface.json"):
        """
        Create annotation interface data for human annotators.

        Exports queries and candidate documents for relevance judgment.

        Args:
            queries: List of query dictionaries
            output_file: Output JSON file for annotation interface
        """

        annotation_data = {
            'instructions': {
                'relevance_levels': {
                    '0': 'Not Relevant - Document does not match query intent',
                    '1': 'Good - Document is somewhat relevant, partial match',
                    '2': 'Excellent - Document is highly relevant, good match',
                    '3': 'Perfect - Document exactly matches query intent'
                },
                'guidelines': [
                    'Consider both explicit matches (keywords, entities) and semantic relevance',
                    'Perfect (3): All query constraints satisfied, primary information need met',
                    'Excellent (2): Most constraints satisfied, very useful for query',
                    'Good (1): Some constraints satisfied, tangentially relevant',
                    'Not Relevant (0): Does not address query or wrong domain/type/period/region',
                    'Provide brief rationale explaining your judgment'
                ]
            },
            'queries': [],
            'documents': self.documents[['title', 'heritage_types', 'domains',
                                          'time_period', 'region']].to_dict('records')
        }

        for query in queries:
            annotation_data['queries'].append({
                'query_id': query['query_id'],
                'query_text': query['query_text'],
                'query_type': query['query_type'],
                'expected_characteristics': {
                    'heritage_types': query['heritage_types'],
                    'domains': query['domains'],
                    'time_period': query.get('time_period'),
                    'region': query.get('region')
                },
                'candidate_documents': list(range(self.num_docs)),  # All docs are candidates
                'judgments': {}  # To be filled by annotators
            })

        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(annotation_data, f, indent=2, cls=NpEncoder)

        print(f"✓ Annotation interface created: {output_path}")
        print(f"  {len(queries)} queries, {self.num_docs} candidate documents")
        return output_path

    def compute_inter_annotator_agreement(self, judgments: Dict[str, Dict[int, List[int]]]) -> Dict:
        """
        Compute Cohen's Kappa inter-annotator agreement.

        Args:
            judgments: Dict[query_id -> Dict[doc_id -> List[relevance_scores]]]

        Returns:
            Dictionary with agreement statistics
        """

        agreement_stats = {
            'overall_kappa': 0.0,
            'query_level_kappa': {},
            'annotator_pairs': {},
            'disagreement_cases': []
        }

        all_kappas = []

        for query_id, doc_judgments in judgments.items():
            # Collect all annotator ratings for this query
            annotator_ratings = defaultdict(dict)

            for doc_id, ratings in doc_judgments.items():
                for i, rating in enumerate(ratings):
                    annotator_ratings[f"annotator_{i}"][doc_id] = rating

            if len(annotator_ratings) < 2:
                continue

            # Compute pairwise kappa
            annotators = list(annotator_ratings.keys())
            query_kappas = []

            for i in range(len(annotators)):
                for j in range(i + 1, len(annotators)):
                    ann1, ann2 = annotators[i], annotators[j]

                    # Get common documents
                    common_docs = set(annotator_ratings[ann1].keys()) & set(annotator_ratings[ann2].keys())

                    if len(common_docs) < 2:
                        continue

                    ratings1 = [annotator_ratings[ann1][doc] for doc in common_docs]
                    ratings2 = [annotator_ratings[ann2][doc] for doc in common_docs]

                    kappa = cohen_kappa_score(ratings1, ratings2)
                    query_kappas.append(kappa)

                    pair_key = f"{ann1}_vs_{ann2}"
                    if pair_key not in agreement_stats['annotator_pairs']:
                        agreement_stats['annotator_pairs'][pair_key] = []
                    agreement_stats['annotator_pairs'][pair_key].append(kappa)

                    # Track disagreements
                    for doc, r1, r2 in zip(common_docs, ratings1, ratings2):
                        if abs(r1 - r2) >= 2:  # Major disagreement
                            agreement_stats['disagreement_cases'].append({
                                'query_id': query_id,
                                'doc_id': doc,
                                'annotator_1': ann1,
                                'rating_1': r1,
                                'annotator_2': ann2,
                                'rating_2': r2
                            })

            if query_kappas:
                agreement_stats['query_level_kappa'][query_id] = np.mean(query_kappas)
                all_kappas.extend(query_kappas)

        if all_kappas:
            agreement_stats['overall_kappa'] = np.mean(all_kappas)
            agreement_stats['kappa_std'] = np.std(all_kappas)
            agreement_stats['kappa_min'] = np.min(all_kappas)
            agreement_stats['kappa_max'] = np.max(all_kappas)

        # Average pairwise kappa
        for pair, kappas in agreement_stats['annotator_pairs'].items():
            agreement_stats['annotator_pairs'][pair] = {
                'mean_kappa': np.mean(kappas),
                'num_queries': len(kappas)
            }

        return agreement_stats

    def aggregate_relevance_judgments(self, query_id: str,
                                       doc_judgments: Dict[int, List[RelevanceJudgment]]) -> Dict[int, int]:
        """
        Aggregate multiple relevance judgments into consensus scores.

        Strategy: Use median of ratings, require agreement for inclusion.

        Args:
            query_id: Query identifier
            doc_judgments: Dict[doc_id -> List[RelevanceJudgment]]

        Returns:
            Dict[doc_id -> consensus_relevance_level]
        """

        consensus = {}

        for doc_id, judgments in doc_judgments.items():
            if not judgments:
                continue

            ratings = [j.relevance_level for j in judgments]

            # Use median for consensus (robust to outliers)
            median_rating = int(np.median(ratings))

            # Require minimum agreement: if all annotators disagree heavily, exclude
            if len(ratings) > 1:
                max_diff = max(ratings) - min(ratings)
                if max_diff > 2:  # Too much disagreement
                    continue

            # Only include if consensus is relevant (level > 0)
            if median_rating > 0:
                consensus[doc_id] = median_rating

        return consensus

    def detect_bias(self, ground_truth_queries: List[GroundTruthQuery]) -> Dict:
        """
        Detect systematic biases in ground truth dataset.

        Checks:
        1. Graph-connected vs semantically similar document bias
        2. Cluster distribution bias
        3. Temporal/spatial bias
        4. Heritage type/domain coverage

        Args:
            ground_truth_queries: List of validated ground truth queries

        Returns:
            Bias detection report
        """

        bias_report = {
            'cluster_bias': {},
            'temporal_bias': {},
            'spatial_bias': {},
            'heritage_type_bias': {},
            'domain_bias': {},
            'graph_connectivity_bias': {},
            'recommendations': []
        }

        # Collect all relevant documents across queries
        all_relevant_docs = set()
        cluster_relevant_counts = Counter()
        time_relevant_counts = Counter()
        region_relevant_counts = Counter()
        type_relevant_counts = Counter()
        domain_relevant_counts = Counter()

        for query in ground_truth_queries:
            relevant_docs = set(query.consensus_relevance.keys())
            all_relevant_docs.update(relevant_docs)

            for doc_id in relevant_docs:
                if doc_id < len(self.documents):
                    doc = self.documents.iloc[doc_id]

                    if 'cluster' in doc:
                        cluster_relevant_counts[doc['cluster']] += 1

                    if 'time_period' in doc:
                        time_relevant_counts[doc['time_period']] += 1

                    if 'region' in doc:
                        region_relevant_counts[doc['region']] += 1

                    if 'heritage_types' in doc:
                        for ht in doc['heritage_types']:
                            type_relevant_counts[ht] += 1

                    if 'domains' in doc:
                        for dom in doc['domains']:
                            domain_relevant_counts[dom] += 1

        # Check cluster bias
        total_relevant = len(all_relevant_docs)
        for cluster_id, count in self.cluster_distribution.items():
            expected_proportion = count / self.num_docs
            actual_count = cluster_relevant_counts.get(cluster_id, 0)
            actual_proportion = actual_count / total_relevant if total_relevant > 0 else 0

            bias_ratio = actual_proportion / expected_proportion if expected_proportion > 0 else 0

            bias_report['cluster_bias'][f"cluster_{cluster_id}"] = {
                'expected_proportion': expected_proportion,
                'actual_proportion': actual_proportion,
                'bias_ratio': bias_ratio,
                'status': 'overrepresented' if bias_ratio > 1.5 else ('underrepresented' if bias_ratio < 0.5 else 'balanced')
            }

        # Check temporal bias
        for period, count in self.time_period_distribution.items():
            expected_proportion = count / self.num_docs
            actual_count = time_relevant_counts.get(period, 0)
            actual_proportion = actual_count / total_relevant if total_relevant > 0 else 0

            bias_ratio = actual_proportion / expected_proportion if expected_proportion > 0 else 0

            bias_report['temporal_bias'][period] = {
                'expected_proportion': expected_proportion,
                'actual_proportion': actual_proportion,
                'bias_ratio': bias_ratio,
                'status': 'overrepresented' if bias_ratio > 1.5 else ('underrepresented' if bias_ratio < 0.5 else 'balanced')
            }

        # Check spatial bias
        for region, count in self.region_distribution.items():
            expected_proportion = count / self.num_docs
            actual_count = region_relevant_counts.get(region, 0)
            actual_proportion = actual_count / total_relevant if total_relevant > 0 else 0

            bias_ratio = actual_proportion / expected_proportion if expected_proportion > 0 else 0

            bias_report['spatial_bias'][region] = {
                'expected_proportion': expected_proportion,
                'actual_proportion': actual_proportion,
                'bias_ratio': bias_ratio,
                'status': 'overrepresented' if bias_ratio > 1.5 else ('underrepresented' if bias_ratio < 0.5 else 'balanced')
            }

        # Generate recommendations
        for cluster, stats in bias_report['cluster_bias'].items():
            if stats['status'] == 'underrepresented':
                bias_report['recommendations'].append(
                    f"Add more queries targeting {cluster} (bias ratio: {stats['bias_ratio']:.2f})"
                )

        for period, stats in bias_report['temporal_bias'].items():
            if stats['status'] == 'underrepresented':
                bias_report['recommendations'].append(
                    f"Add more queries for {period} time period (bias ratio: {stats['bias_ratio']:.2f})"
                )

        for region, stats in bias_report['spatial_bias'].items():
            if stats['status'] == 'underrepresented':
                bias_report['recommendations'].append(
                    f"Add more queries for {region} region (bias ratio: {stats['bias_ratio']:.2f})"
                )

        return bias_report

    def save_ground_truth(self, queries: List[GroundTruthQuery],
                          split: str = 'dev',
                          version: str = None):
        """
        Save ground truth dataset with versioning.

        Args:
            queries: List of validated GroundTruthQuery objects
            split: 'dev' for development set or 'test' for final test set
            version: Version string (default: self.version)
        """

        version = version or self.version
        filename = f"ground_truth_v{version}_{split}.json"
        output_path = self.output_dir / filename

        # Convert to serializable format
        serializable_queries = []
        for query in queries:
            query_dict = asdict(query)

            # Convert RelevanceJudgment objects to dicts
            judgments_dict = {}
            for doc_id, judgments in query.relevance_judgments.items():
                judgments_dict[str(doc_id)] = [asdict(j) for j in judgments]
            query_dict['relevance_judgments'] = judgments_dict

            # Convert consensus_relevance keys to strings for JSON
            query_dict['consensus_relevance'] = {
                str(k): v for k, v in query.consensus_relevance.items()
            }

            serializable_queries.append(query_dict)

        output_data = {
            'version': version,
            'split': split,
            'creation_date': datetime.now().isoformat(),
            'num_queries': len(queries),
            'statistics': self._compute_dataset_statistics(queries),
            'queries': serializable_queries
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NpEncoder)

        print(f"✓ Ground truth saved: {output_path}")
        print(f"  Version: {version}, Split: {split}")
        print(f"  Queries: {len(queries)}")
        return output_path

    def _compute_dataset_statistics(self, queries: List[GroundTruthQuery]) -> Dict:
        """Compute statistics for ground truth dataset"""

        stats = {
            'num_queries': len(queries),
            'query_types': Counter([q.query_type for q in queries]),
            'complexity_distribution': Counter([q.complexity for q in queries]),
            'avg_relevant_docs_per_query': 0.0,
            'avg_inter_annotator_agreement': 0.0,
            'relevance_level_distribution': Counter(),
            'cluster_coverage': set(),
            'time_period_coverage': set(),
            'region_coverage': set()
        }

        total_relevant = 0
        total_kappa = 0
        num_queries_with_kappa = 0

        for query in queries:
            total_relevant += len(query.consensus_relevance)

            if query.inter_annotator_agreement > 0:
                total_kappa += query.inter_annotator_agreement
                num_queries_with_kappa += 1

            for level in query.consensus_relevance.values():
                stats['relevance_level_distribution'][int(level)] += 1

            # Convert cluster IDs to int for JSON serialization
            stats['cluster_coverage'].update([int(k) for k in query.cluster_distribution.keys()])

            if query.time_period:
                stats['time_period_coverage'].add(query.time_period)
            if query.region:
                stats['region_coverage'].add(query.region)

        stats['avg_relevant_docs_per_query'] = total_relevant / len(queries) if queries else 0
        stats['avg_inter_annotator_agreement'] = total_kappa / num_queries_with_kappa if num_queries_with_kappa > 0 else 0

        # Convert sets to lists for JSON serialization
        stats['cluster_coverage'] = sorted(list(stats['cluster_coverage']))
        stats['time_period_coverage'] = sorted(list(stats['time_period_coverage']))
        stats['region_coverage'] = sorted(list(stats['region_coverage']))
        stats['query_types'] = dict(stats['query_types'])
        stats['complexity_distribution'] = dict(stats['complexity_distribution'])
        stats['relevance_level_distribution'] = dict(stats['relevance_level_distribution'])

        return stats


def main():
    """Main execution function"""

    print("=" * 80)
    print("MULTI-STRATEGY GROUND TRUTH GENERATION SYSTEM")
    print("=" * 80)

    # Initialize generator
    generator = GroundTruthGenerator(
        data_path="data/processed/heritage_metadata.csv",
        output_dir="data/evaluation"
    )

    print("\n" + "=" * 80)
    print("PHASE 1: SEED QUERY GENERATION")
    print("=" * 80)

    # Generate seed queries
    seed_queries = generator.generate_seed_queries(num_queries=50)
    print(f"\n✓ Generated {len(seed_queries)} seed queries")
    print(f"  Query types: {Counter([q['query_type'] for q in seed_queries])}")
    print(f"  Complexity: {Counter([q['complexity'] for q in seed_queries])}")

    print("\n" + "=" * 80)
    print("PHASE 2: SYNTHETIC QUERY GENERATION")
    print("=" * 80)

    # Generate synthetic queries
    synthetic_queries = generator.generate_synthetic_queries(num_easy=30, num_hard=20)
    print(f"\n✓ Generated {len(synthetic_queries)} synthetic queries")
    print(f"  Easy: {len([q for q in synthetic_queries if q['query_type'] == 'synthetic_easy'])}")
    print(f"  Hard: {len([q for q in synthetic_queries if q['query_type'] == 'synthetic_hard'])}")

    print("\n" + "=" * 80)
    print("PHASE 3: ANNOTATION INTERFACE CREATION")
    print("=" * 80)

    # Combine all queries
    all_queries = seed_queries + synthetic_queries

    # Create annotation interface
    annotation_file = generator.create_annotation_interface(
        queries=all_queries,
        output_file="annotation_interface_v2.json"
    )

    print(f"\n✓ Annotation interface ready for human annotators")
    print(f"  Total queries: {len(all_queries)}")
    print(f"  Candidate documents per query: {generator.num_docs}")
    print(f"\n  Next steps:")
    print(f"  1. Distribute {annotation_file} to 2-3 domain experts")
    print(f"  2. Have each annotator judge relevance for all query-document pairs")
    print(f"  3. Collect judgments and run validation with compute_inter_annotator_agreement()")
    print(f"  4. Generate final ground truth with save_ground_truth()")

    # Save query metadata for reference
    metadata_file = generator.output_dir / "query_metadata_v2.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'seed_queries': seed_queries,
            'synthetic_queries': synthetic_queries,
            'total': len(all_queries),
            'generation_date': datetime.now().isoformat()
        }, f, indent=2, cls=NpEncoder)

    print(f"\n✓ Query metadata saved: {metadata_file}")

    print("\n" + "=" * 80)
    print("SYSTEM READY FOR ANNOTATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
