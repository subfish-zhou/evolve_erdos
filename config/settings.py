import os
from dotenv import load_dotenv

load_dotenv()

# Helper function to get env var and handle empty strings
def _get_env_or_none(key: str, default: str = None):
    """Get environment variable, return None if empty string or not set."""
    value = os.getenv(key, default)
    return value if value and value.strip() else (default if default else None)


def _get_int_env(key: str, default: int) -> int:
    """Get an integer environment variable with a safe fallback."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get a float environment variable with a safe fallback."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default

# LLM Configuration
FLASH_API_KEY = _get_env_or_none("FLASH_API_KEY")
FLASH_BASE_URL = _get_env_or_none("FLASH_BASE_URL")
FLASH_MODEL = _get_env_or_none("FLASH_MODEL")

# PRO_API_KEY = os.getenv("PRO_API_KEY")
# PRO_BASE_URL = os.getenv("PRO_BASE_URL", None)
# PRO_MODEL = os.getenv("PRO_MODEL")

EVALUATION_API_KEY = _get_env_or_none("EVALUATION_API_KEY")
EVALUATION_BASE_URL = _get_env_or_none("EVALUATION_BASE_URL")
EVALUATION_MODEL = _get_env_or_none("EVALUATION_MODEL")

# LiteLLM Configuration
LITELLM_DEFAULT_MODEL = os.getenv("LITELLM_DEFAULT_MODEL", "gpt-3.5-turbo") or "gpt-3.5-turbo"
LITELLM_DEFAULT_BASE_URL = _get_env_or_none("LITELLM_DEFAULT_BASE_URL")
LITELLM_MAX_TOKENS = _get_env_or_none("LITELLM_MAX_TOKENS")
LITELLM_TEMPERATURE = _get_env_or_none("LITELLM_TEMPERATURE")
LITELLM_TOP_P = _get_env_or_none("LITELLM_TOP_P")
LITELLM_TOP_K = _get_env_or_none("LITELLM_TOP_K")

# Azure OpenAI Configuration
USE_AZURE_OPENAI = os.getenv("USE_AZURE_OPENAI", "False").lower() == "true"
AZURE_ENDPOINT_URL = os.getenv("ENDPOINT_URL", "https://ai4mtest1.openai.azure.com/")
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-5-mini")
AZURE_MANAGED_IDENTITY_CLIENT_ID = os.getenv(
    "AZURE_MANAGED_IDENTITY_CLIENT_ID",
    "7e0d39de-9cb1-4585-85af-1e82ea00b36d"
)
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_HTTP_TIMEOUT_SECONDS = _get_float_env("AZURE_HTTP_TIMEOUT_SECONDS", 600.0)
AZURE_MAX_RETRIES = _get_int_env("AZURE_MAX_RETRIES", 3)
AZURE_API_REQUEST_LOG = os.getenv("AZURE_API_REQUEST_LOG", "azure_api_requests.log")

# Specific model names for strategic use (can be same as LITELLM_DEFAULT_MODEL if only one is used)
# 如果启用了 Azure OpenAI，默认使用 Azure 部署名称
if USE_AZURE_OPENAI:
    DEFAULT_PRIMARY_MODEL = AZURE_DEPLOYMENT_NAME
    DEFAULT_SECONDARY_MODEL = AZURE_DEPLOYMENT_NAME
else:
    DEFAULT_PRIMARY_MODEL = LITELLM_DEFAULT_MODEL
    DEFAULT_SECONDARY_MODEL = FLASH_MODEL if FLASH_MODEL else LITELLM_DEFAULT_MODEL

# Get model names from env, but ensure they are not empty strings
_primary_from_env = _get_env_or_none("LLM_PRIMARY_MODEL")
_secondary_from_env = _get_env_or_none("LLM_SECONDARY_MODEL")

LLM_PRIMARY_MODEL = _primary_from_env if _primary_from_env else DEFAULT_PRIMARY_MODEL
LLM_SECONDARY_MODEL = _secondary_from_env if _secondary_from_env else DEFAULT_SECONDARY_MODEL

# Validate that model names are not None or empty
if not LLM_PRIMARY_MODEL or not LLM_PRIMARY_MODEL.strip():
    raise ValueError("LLM_PRIMARY_MODEL must be set to a valid model name. Check your .env file or environment variables.")
if not LLM_SECONDARY_MODEL or not LLM_SECONDARY_MODEL.strip():
    raise ValueError("LLM_SECONDARY_MODEL must be set to a valid model name. Check your .env file or environment variables.")

# if not PRO_API_KEY:
#     print("Warning: PRO_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
#     PRO_API_KEY = "Your API key"

# Evolutionary Algorithm Settings
POPULATION_SIZE = 6
GENERATIONS = 2
# Threshold for switching to bug-fix prompt
# If a program has errors and its correctness score is below this, a bug-fix prompt will be used.
BUG_FIX_CORRECTNESS_THRESHOLD = float(os.getenv("BUG_FIX_CORRECTNESS_THRESHOLD", "0.05"))
# Threshold for using the primary (potentially more powerful/expensive) LLM for mutation
HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM = float(os.getenv("HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM", "0.4"))
ELITISM_COUNT = 0
MUTATION_RATE = 0.9
CROSSOVER_RATE = 0.1

# Island Model Settings
NUM_ISLANDS = 6  # Number of subpopulations
MIGRATION_INTERVAL = 6  # Number of generations between migrations
ISLAND_POPULATION_SIZE = POPULATION_SIZE // NUM_ISLANDS  # Programs per island
MIN_ISLAND_SIZE = 2  # Minimum number of programs per island
MIGRATION_RATE = 0.2  # Rate at which programs migrate between islands

# Debug Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
EVALUATION_TIMEOUT_SECONDS = 800

# Docker Execution Settings
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME", "code-evaluator:latest")
DOCKER_NETWORK_DISABLED = os.getenv("DOCKER_NETWORK_DISABLED", "True").lower() == "true"

DATABASE_TYPE = "json"
DATABASE_PATH = "program_database.json"

# Logging Configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_FILE = "alpha_evolve.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10

RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080"

def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_llm_model(model_type="default"):
    if model_type == "default":
        return LITELLM_DEFAULT_MODEL
    elif model_type == "flash":
        # Assuming FLASH_MODEL might still be a specific, different model.
        # If FLASH_MODEL is also meant to be covered by litellm's general handling,
        # this could also return LITELLM_DEFAULT_MODEL or a specific flash model string.
        # For now, keep FLASH_MODEL if it's distinct.
        return FLASH_MODEL if FLASH_MODEL else LITELLM_DEFAULT_MODEL # Return default if FLASH_MODEL is not set
    # Fallback for any other model_type not explicitly handled
    return LITELLM_DEFAULT_MODEL

                                 
