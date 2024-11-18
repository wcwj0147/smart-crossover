import os
import pickle
from typing import Optional, List, Any, Union

import gurobipy
from matplotlib import pyplot as plt

from smart_crossover import get_project_root
from smart_crossover.formats import GeneralLP
from smart_crossover.solver_caller.gurobi import GrbCaller


class FileHandler:
    """
    A class representing a filehandler, who reads, writes, and transfer optimization models using Gurobi.

    Attributes:
        models (list[gurobipy.Model]): currently saved models.
        grbCaller (GrbCaller): a gurobi caller to presolve or transfer models.

    """
    models: List[gurobipy.Model]
    grbCaller: GrbCaller

    def __init__(self, models: Optional[List[gurobipy.Model]] = None) -> None:
        self.models = models
        self.grbCaller = GrbCaller()

    def read_models_from_files(self, path: str) -> None:
        """
        Clear saved models and read new models from .mps or .lp files.

        Args:
            path: the path of aimed files. The method reads all acceptable files from the path.

        """
        self.models = []
        files: List[str] = [f for f in os.listdir(path) if f.endswith('.mps') or f.endswith('.lp')]
        for file in files:
            self.grbCaller.read_model_from_file(path + "/" + file)
            base_name, _ = os.path.splitext(file)
            self.grbCaller.model.ModelName = base_name  # to name the model in a uniform way.
            self.grbCaller.model.update()
            self.models.append(self.grbCaller.model)

    def read_mcf_bm_from_files(self) -> None:
        """ Read mcf benchmark models from location: data/network. """
        path = str(get_project_root() / f"data/network")
        self.read_models_from_files(path)

    def read_lp_from_files(self, file_type: str = "all") -> None:
        """
        Read original or presolved benchmark LP problems.

        Args:
            file_type: choose from {"all", "standard", "presolved"}

        """
        path = str(get_project_root() / f"data/lp/{file_type}")
        self.read_models_from_files(path)

    def write_presolved_models(self, path: str = str(get_project_root() / "data/lp/presolved")) -> None:
        """
        Presolve models using Gurobi and write them to the given path.

        Args:
            path: the path to save presolved models.

        """
        for model in self.models:
            print(f"Consider model: {model.ModelName}")
            presolved_model = model.relax().presolve()
            presolved_model.ModelName = model.ModelName
            presolved_model.write(path + f"/{model.ModelName}.mps")

    def get_models_report(self, *args: Union[List[str], str]) -> str:
        """ Get a report string of models. """
        report_str = "\n****** Reports about models' info ******\n"
        model_names = [name for arg in args for name in (arg if isinstance(arg, list) else [arg])]
        models = self.models if not model_names else [self.get_model_by_name(name) for name in model_names]
        for model in models:
            self.grbCaller.read_model(model)
            report_str += self.grbCaller.get_model_report()
            report_str += "\n"
        return report_str

    def get_model_by_name(self, model_name: str) -> gurobipy.Model:
        """ Get a model by its name. """
        for model in self.models:
            if model.ModelName == model_name:
                return model
        raise ValueError(f"Model {model_name} not found.")

    def get_lp_by_name(self, model_name: str) -> GeneralLP:
        """ Get a model by its name. """
        model = self.get_model_by_name(model_name)
        self.grbCaller.read_model(model)
        return self.grbCaller.return_genlp()


def read_results_from_pickle(path: str) -> Any:
    """ Read results from pickle file."""
    with open(get_project_root() / "results" / path, 'rb') as infile:
        results = pickle.load(infile)
    return results


def write_results_to_pickle(results: Any, path: str) -> None:
    """ Write results to pickle file."""
    with open(get_project_root() / "results" / path, 'wb') as outfile:
        pickle.dump(results, outfile)
