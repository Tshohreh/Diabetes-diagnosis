from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# بارگذاری داده‌ها
data = pd.read_csv('DATASET\diabetes.csv')

# تقسیم داده‌ها
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# پیش‌پردازش: مقیاس‌بندی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ایجاد و آموزش مدل SVM
model = SVC(kernel='rbf', C=1, gamma='scale')  # RBF kernel
model.fit(X_train, y_train)

# پیش‌بینی
y_pred = model.predict(X_test)

# ارزیابی مدل
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

linear_model = SVC(kernel='linear', probability=True)
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
