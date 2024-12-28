import hashlib
import os
import base64
import json
import cirq
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# Function to generate quantum random numbers using Cirq
def quantum_random_number(bit_count):
    """Generates a quantum random number with the specified bit count using Cirq."""
    qubits = [cirq.LineQubit(i) for i in range(bit_count)]
    circuit = cirq.Circuit()

    # Apply Hadamard gates to all qubits to create superposition
    circuit.append(cirq.H.on_each(qubits))
    circuit.append(cirq.measure(*qubits, key='result'))

    # Simulate the quantum circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    measurements = result.measurements['result'][0]  # Single repetition
    return int("".join(map(str, measurements)), 2)


def train_integrity_models(binary_key):
    """Train individual anomaly detection models to check integrity."""
    binary_key = np.array(list(map(int, binary_key))).reshape(-1, 1)

    # Train IsolationForest
    iforest = IsolationForest(contamination=0.05, random_state=42)
    iforest.fit(binary_key)

    # Train LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof.fit(binary_key)

    return iforest, lof


def check_integrity(binary_key, models):
    """Check the integrity of the binary key using trained models."""
    binary_key = np.array(list(map(int, binary_key))).reshape(-1, 1)
    iforest, lof = models

    # Validate using both models
    iforest_prediction = iforest.predict(binary_key)
    lof_prediction = lof.fit_predict(binary_key)

    # Integrity passes if both models agree the key is valid (label 1)
    return all(pred == 1 for pred in iforest_prediction) and all(pred == 1 for pred in lof_prediction)


def login_password_to_hash(
    login, password, salt=None, iterations=100000, quantum_bits=256
):
    """Generate a secure hash using login, password, and quantum random entropy."""
    if salt is None:
        salt = os.urandom(16)  # Generate a 16-byte random salt

    # Generate a quantum random number and append it to the login and password
    quantum_entropy = quantum_random_number(quantum_bits)
    binary_key = bin(quantum_entropy)[2:].zfill(quantum_bits)

    # Train anomaly detection models
    models = train_integrity_models(binary_key)

    combined_data = f"{login}:{password}:{quantum_entropy}"

    # Derive the hash using PBKDF2-HMAC-SHA256
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        combined_data.encode(),
        salt,
        iterations
    )

    # Create HMAC for integrity check
    hmac_key = os.urandom(16)
    hmac = hashlib.pbkdf2_hmac(
        'sha256',
        hash_bytes,
        hmac_key,
        iterations
    )

    # Combine and encode data
    output_data = {
        "salt": salt.hex(),
        "hash": hash_bytes.hex(),
        "quantum_entropy": quantum_entropy,
        "iterations": iterations,
        "hmac": hmac.hex(),
        "hmac_key": base64.b64encode(hmac_key).decode(),
        "binary_key": binary_key,
    }

    return base64.b64encode(json.dumps(output_data).encode()).decode(), models


def verify_login_password(login, password, stored_data, models):
    """Verify the login and password against stored hash data."""
    # Decode and load the stored data
    decoded_data = json.loads(base64.b64decode(stored_data).decode())
    salt = bytes.fromhex(decoded_data["salt"])
    quantum_entropy = decoded_data["quantum_entropy"]
    iterations = decoded_data["iterations"]
    original_hash = bytes.fromhex(decoded_data["hash"])
    hmac_key = base64.b64decode(decoded_data["hmac_key"])
    original_hmac = bytes.fromhex(decoded_data["hmac"])
    binary_key = decoded_data["binary_key"]

    # Check key integrity
    if not check_integrity(binary_key, models):
        return False

    # Recompute hash
    combined_data = f"{login}:{password}:{quantum_entropy}"
    recomputed_hash = hashlib.pbkdf2_hmac(
        'sha256',
        combined_data.encode(),
        salt,
        iterations
    )

    # Verify hash and HMAC
    if recomputed_hash != original_hash:
        return False

    recomputed_hmac = hashlib.pbkdf2_hmac(
        'sha256',
        recomputed_hash,
        hmac_key,
        iterations
    )
    return recomputed_hmac == original_hmac


# Example usage
login = "User123"
password = "SecurePassword123!"

# Generate hash and train models
stored_data, models = login_password_to_hash(login, password)

print("Stored Data (Encoded):", stored_data)

# Verify credentials
is_valid = verify_login_password(login, password, stored_data, models)
print("Is Valid Login:", is_valid)