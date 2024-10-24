import requests
import json

def get_pokemon_evolution_chain(evolution_chain_url):
    try:
        response = requests.get(evolution_chain_url)
        response.raise_for_status()
        evolution_chain_data = response.json()
        return evolution_chain_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def build_nested_evolutions(chain):
    current_species = {"name": chain['species']['name'], "variations": []}

    if chain['evolves_to']:
        next_evolution = chain['evolves_to'][0]
        current_species['variations'].append(build_nested_evolutions(next_evolution))

    return current_species

def get_pokemon_species(pokemon_name):
    url = f"https://challenges.hackajob.co/pokeapi/api/v2/pokemon-species/{pokemon_name}/"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        pokemon_data = response.json()
        return pokemon_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")


if __name__ == "__main__":
    pokemon_name = "butterfree"
    pokemon_species = get_pokemon_species(pokemon_name)
    evolution_chain_url = pokemon_species["evolution_chain"]["url"]
 
    evolution_chain_data = get_pokemon_evolution_chain(evolution_chain_url)

    result = build_nested_evolutions(evolution_chain_data["chain"])
    print(json.dumps(result, indent=4))
