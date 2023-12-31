from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


def regression(x, y, test_size=0.2, random_state=0, scale_data=False):
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Optional: Scale data
    if scale_data:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Create and fit the model
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    return model, y_pred


def feature_selection_rfe(x, y, n_features_to_select=5):
    # Scaling the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Create a logistic regression classifier with increased max_iter
    logreg = LogisticRegression(max_iter=1000)

    # RFE model
    rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(x_train, y_train)

    # Summarize the selection of the attributes
    print('Selected features:', list(x.columns[rfe.support_]))
    print('Feature Ranking:', rfe.ranking_)

    return x.columns[rfe.support_]
