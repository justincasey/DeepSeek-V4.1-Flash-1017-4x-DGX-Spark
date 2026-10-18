"""servebench: steady-state load and precision measurements for OpenAI-compatible LLM
servers.

Standard library only, so it runs from any host Python against any server. Nothing here
imports the cluster tooling around it; a site calls it as a command:

    python3 -m servebench run URL --mode closed --concurrency 16 --output run.json
    python3 -m servebench compare base.json candidate.json

See README.md for the methodology.
"""

__version__ = "0.1.0"
SCHEMA = "servebench/1"
