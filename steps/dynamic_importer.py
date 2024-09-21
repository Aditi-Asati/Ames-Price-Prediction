from zenml import step
import pandas as pd

@step
def dynamic_importer() -> str:
    data = {}

    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")

    return json_data