#!/usr/bin/env bash
source .venv/bin/activate
jupyter lab --IdentityProvider.token MY_TOKEN --ip 0.0.0.0 --no-browser
