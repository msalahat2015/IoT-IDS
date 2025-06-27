import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from ..base.base_algorithm import BaseAlgorithm  # type: ignore

class RNNModel(BaseAlgorithm):
    def __init__(self, params=None):
        self.params = params or {}
        self.model = self._build_model()

    def _build_model(self):
        p = self.params
        raw_shape = p.get('input_shape')
        if not raw_shape or len(raw_shape) != 2:
            raise ValueError("input_shape must be a list [timesteps, features]")
        input_shape = tuple(raw_shape)  # e.g., (1, 29)

        num_classes = p.get('num_classes')
        if num_classes is None:
            raise ValueError("'num_classes' must be defined")

        model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(
                units=p.get('units', 64),
                activation=p.get('activation', 'tanh'),
                kernel_initializer=p.get('kernel_initializer', 'glorot_uniform'),
                recurrent_initializer=p.get('recurrent_initializer', 'orthogonal'),
                dropout=p.get('dropout', 0.0),
                recurrent_dropout=p.get('recurrent_dropout', 0.0),
                return_sequences=False
            ),
            Dropout(p.get('dropout', 0.0)),
            Dense(num_classes, activation='softmax')
        ])

        optimizer_param = p.get('optimizer', 'adam')
        if isinstance(optimizer_param, str) and optimizer_param.lower() == 'adam':
            optimizer = Adam(learning_rate=p.get('learning_rate', 0.001))
        else:
            optimizer = optimizer_param

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=p.get('metrics', ['accuracy'])
        )
        return model

    def _ensure_3d(self, X):
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)
        return X

    def train(self, X, y):
        X = self._ensure_3d(X)
        num_classes = self.params['num_classes']

        # Convert y to class indices if it is one-hot encoded
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        y = to_categorical(y, num_classes=num_classes)

        p = self.params
        if p.get('validation_split', 0.0) > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=p['validation_split'], random_state=42
            )
            self.model.fit(
                X_train, y_train,
                epochs=p.get('epochs', 10),
                batch_size=p.get('batch_size', 32),
                validation_data=(X_val, y_val)
            )
        else:
            self.model.fit(
                X, y,
                epochs=p.get('epochs', 10),
                batch_size=p.get('batch_size', 32)
            )

    def predict(self, X):
        X = self._ensure_3d(X)
        return self.model.predict(X)

    def evaluate(self, X, y):
        X = self._ensure_3d(X)
        num_classes = self.params['num_classes']

        # Convert y to class indices if one-hot encoded
        if y.ndim > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y

        y_pred_prob = self.model.predict(X)
        y_pred = np.argmax(y_pred_prob, axis=1)

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred),
        }

        try:
            if num_classes > 2:
                results["auc_roc"] = roc_auc_score(
                    to_categorical(y_true, num_classes=num_classes),
                    y_pred_prob,
                    multi_class='ovr'
                )
            else:
                results["auc_roc"] = roc_auc_score(y_true, y_pred_prob[:, 1])
        except Exception as e:
            results["auc_roc"] = f"Error computing AUC ROC: {e}"

        return results

    def get_model(self):
        return self.model
