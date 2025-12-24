import sys

# Provide clearer error messages when required third-party packages are missing.
try:
    import pandas as pd
except Exception as e:
    raise ImportError(
        "Missing dependency 'pandas'. Install it with:\n"
        "  & \"C:/Program Files/Python311/python.exe\" -m pip install pandas\n"
        "or\n"
        "  py -3 -m pip install pandas\n"
        "or\n"
        "  python -m pip install pandas\n"
        "If you're using a virtual environment, activate it first."
    ) from e

try:
    import yaml
except Exception as e:
    raise ImportError(
        "Missing dependency 'pyyaml' (imported as 'yaml'). Install it with:\n"
        "  & \"C:/Program Files/Python311/python.exe\" -m pip install pyyaml\n"
        "or\n"
        "  py -3 -m pip install pyyaml\n"
        "or\n"
        "  python -m pip install pyyaml\n"
    ) from e

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def load_schema(schema_path):
    with open(schema_path, "r") as file:
        return yaml.safe_load(file)

def extract_column_names(schema):
    columns = schema.get("columns", [])
    return [list(col.keys())[0] for col in columns]

class DataValidator:
    def __init__(self, data, schema):
        self.data = data
        self.schema = schema
        self.results = {}

    def validate_num_columns(self):
        expected = len(extract_column_names(self.schema))
        self.results["num_columns"] = (len(self.data.columns) == expected, expected, len(self.data.columns))

    def validate_column_names(self):
        expected = set(extract_column_names(self.schema))
        self.results["column_names"] = (set(self.data.columns) == expected, list(expected), list(self.data.columns))

    def validate(self):
        self.validate_num_columns()
        self.validate_column_names()

    def summary(self):
        for check, result in self.results.items():
            status, expected, found = result
            if status:
                print(f"{check} check passed.")
            else:
                print(f"{check} check failed. Expected: {expected}, Found: {found}")
