"""Allow `python3 -m helix <subcommand> ...`."""

import sys

from helix.cli import main

if __name__ == "__main__":
    sys.exit(main())
