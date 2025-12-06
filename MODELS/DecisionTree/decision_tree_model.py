from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth=20, min_samples_leaf=10, random_state=42):
    """
    Entrena un DecisionTreeClassifier con los parámetros dados y devuelve el modelo entrenado.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
