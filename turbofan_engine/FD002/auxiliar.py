from pathlib import Path
import sys
import glob

def init()-> str:
    """Informa ao usuário que o módulo foi importado com sucesso.

    Returns
    -------
    str
        Mensagem de importação.
    """
    module_name = Path(__file__).stem 
    return (f"Módulo {module_name} importado com sucesso.")

def search_module(module_name: str) -> Path:
    path_search = Path(__file__)
    was_found = False
    while not was_found:
        path_search = path_search.parent
        path_config_file = glob.glob(str(path_search) + f"/**/__base{module_name}__",
                                     recursive = True)
        len_config_file = len(path_config_file)
        if (len_config_file != 0) and \
           (not len_config_file > 1) and \
           (f'__base{module_name}__'  in path_config_file[0]):
            return Path(path_config_file[0]).parent
    raise (f"Module {module_name} not found.")

module_name = ""# Path(__file__).stem 
path_module = search_module(module_name)
sys.path.append(str(path_module))


if __name__ == '__main__':
    print("Imported")
