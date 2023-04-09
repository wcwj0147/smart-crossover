import os
from typing import Optional

import gurobipy

from smart_crossover import get_project_root
from smart_crossover.solver_caller.gurobi import GrbCaller


# Todo: remove this class. It seems that we only need some functions.
class FileHandler:
    """
    A class representing a filehandler, who reads, writes, and transfer optimization models using Gurobi.

    Attributes:
        models (list[gurobipy.Model]): currently saved models.
        grbCaller (GrbCaller): a gurobi caller to presolve or transfer models.

    """
    models: list[gurobipy.Model]
    grbCaller: GrbCaller

    def __init__(self, models: Optional[list[gurobipy.Model]] = None) -> None:
        self.models = models
        self.grbCaller = GrbCaller()

    def read_models_from_files(self, path: str) -> None:
        """
        Clear saved models and read new models from .mps or .lp files.

        Args:
            path: the path of aimed files. The method reads all acceptable files from the path.

        """
        self.models = []
        files: list[str] = [f for f in os.listdir(path) if f.endswith('.mps') or f.endswith('.lp')]
        for file in files:
            self.grbCaller.read_model_from_file(path + "/" + file)
            base_name, _ = os.path.splitext(file)
            self.grbCaller.model.ModelName = base_name  # to name the model in a uniform way.
            self.grbCaller.model.update()
            self.models.append(self.grbCaller.model)

    def read_mcf_bm_from_files(self) -> None:
        """ Read mcf benchmark models."""
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
        """ Presolve saved models using Gurobi and write them to the given path."""
        for model in self.models:
            print(f"Consider model: {model.ModelName}")
            presolved_model = model.presolve()
            presolved_model.ModelName = model.ModelName
            presolved_model.write(path + f"/{model.ModelName}.mps")

    def get_models_report(self) -> str:
        """ Get a report string of models. """
        report_str = "\n****** Reports about models' info ******\n"
        for model in self.models:
            self.grbCaller.read_model(model)
            report_str += self.grbCaller.get_model_report()
            report_str += "\n"
        return report_str
