import datetime

# Features
features = {
    "categorical_variables": ["Geography", "Gender"],
    "numerical_variables": [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ],
    "binary_variables": ["HasCrCard", "IsActiveMember"],
}

# Directories
directories = {
    "log_dir": "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    "model_dir": "models/",
}
