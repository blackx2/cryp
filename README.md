# 🔐 Quantum-Enhanced Password Hashing and Integrity Verification

This project demonstrates a secure login and password hashing system that incorporates **quantum entropy**, **anomaly detection**, and **HMAC integrity checks**. It uses simulated quantum randomness via [Cirq](https://github.com/quantumlib/Cirq) and applies machine learning models to validate the integrity of cryptographic keys.

---

## 🚀 Features

- ✅ Quantum-generated entropy using Cirq simulator  
- ✅ Password hashing with PBKDF2-HMAC-SHA256  
- ✅ Anomaly detection with Isolation Forest and Local Outlier Factor  
- ✅ HMAC integrity check for tamper resistance  
- ✅ Base64-encoded JSON structure for storage and reproducibility  

---

## 📦 Technologies Used

- [Cirq](https://quantumai.google/cirq) – Quantum circuit simulation  
- [scikit-learn](https://scikit-learn.org/) – Machine learning models for integrity checks  
- [NumPy](https://numpy.org/) – Numerical operations  
- [hashlib / base64 / json / os] – Python built-ins for security and data encoding  

---

## 🔧 How It Works

1. **Quantum Entropy Generation**  
   - Generates `N`-bit random number from a Hadamard-applied quantum circuit.

2. **Anomaly Detection**  
   - Trains `IsolationForest` and `LocalOutlierFactor` models to verify randomness integrity.

3. **Secure Hashing**  
   - Combines login, password, and quantum entropy.
   - Applies PBKDF2 with SHA-256 for secure hash derivation.
   - Adds HMAC to detect tampering.

4. **Verification**  
   - Validates the entropy integrity using pre-trained models.
   - Recalculates the hash and HMAC to verify credentials.

---

## 🧪 Example Usage

```python
login = "User123"
password = "SecurePassword123!"

# Generate hash and models
stored_data, models = login_password_to_hash(login, password)

# Verify
is_valid = verify_login_password(login, password, stored_data, models)
print("Is Valid Login:", is_valid)
