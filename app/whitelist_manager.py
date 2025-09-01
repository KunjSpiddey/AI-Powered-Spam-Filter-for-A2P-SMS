import yaml
from pathlib import Path
from typing import List

WHITELIST_FILE = Path("config/whitelist.yml")

def load_whitelist() -> dict:
    """Load whitelist from YAML file."""
    with open(WHITELIST_FILE, "r") as f:
        data = yaml.safe_load(f) or {}
    return {
        "domains": data.get("domains", []),
        "phrases": data.get("phrases", [])
    }

def save_whitelist(domains: List[str], phrases: List[str]):
    """Save whitelist back to YAML file."""
    with open(WHITELIST_FILE, "w") as f:
        yaml.dump({"domains": domains, "phrases": phrases}, f)

def add_domain(domain: str):
    wl = load_whitelist()
    if domain not in wl["domains"]:
        wl["domains"].append(domain)
    save_whitelist(wl["domains"], wl["phrases"])

def remove_domain(domain: str):
    wl = load_whitelist()
    if domain in wl["domains"]:
        wl["domains"].remove(domain)
    save_whitelist(wl["domains"], wl["phrases"])

def add_phrase(phrase: str):
    wl = load_whitelist()
    if phrase not in wl["phrases"]:
        wl["phrases"].append(phrase)
    save_whitelist(wl["domains"], wl["phrases"])

def remove_phrase(phrase: str):
    wl = load_whitelist()
    if phrase in wl["phrases"]:
        wl["phrases"].remove(phrase)
    save_whitelist(wl["domains"], wl["phrases"])
