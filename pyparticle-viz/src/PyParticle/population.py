from typing import List, Dict

class Population:
    def __init__(self, species: List[str], parameters: Dict):
        self.species = species
        self.parameters = parameters
        self.data = {}

    def build_population(self):
        # Logic to build the population based on species and parameters
        for species in self.species:
            self.data[species] = self._initialize_species_data(species)

    def _initialize_species_data(self, species: str):
        # Initialize data for a specific species
        return {
            'density': self.parameters.get('density', 0),
            'size_distribution': [],
            'other_metrics': {}
        }

    def update_population(self, new_data: Dict):
        # Logic to update the population with new data
        for species, data in new_data.items():
            if species in self.data:
                self.data[species].update(data)

    def get_population_data(self):
        return self.data

    def clear_population(self):
        self.data.clear()