import cirq
import numpy as np
from sklearn.ensemble import IsolationForest
import hashlib
import oqs
import struct
from pprint import pprint

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

# Function to check key quality using ML
def check_key_quality(key, bit_count):
    """Checks the randomness quality of the key using ML."""
    # Convert key into a binary array
    binary_key = np.array([int(bit) for bit in f"{key:0{bit_count}b}"])

    # Use an Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.05, random_state=42)
    reshaped_key = binary_key.reshape(-1, 1)  # Shape needed for the model
    predictions = model.fit_predict(reshaped_key)

    # If anomalies are detected, return False
    if np.any(predictions == -1):
        return False
    return True

# Generate a quantum-derived random seed
def quantum_seeded_key(byte_count=32, max_attempts=10):
    """Generates a quantum-derived random key, ensuring quality."""
    bit_count = byte_count * 8  # Convert byte count to bit count
    for attempt in range(max_attempts):
        # Generate a random number using quantum randomness
        random_seed = quantum_random_number(bit_count)  # Get a quantum random number with sufficient bits

        # Convert the quantum seed to bytes
        random_seed_bytes = random_seed.to_bytes(byte_count, byteorder='little')

        # Check the quality of the generated key
        if check_key_quality(random_seed, bit_count):
            print(f"Key accepted after {attempt + 1} attempts.")
            return random_seed_bytes
        print(f"Weak key detected, regenerating... (Attempt {attempt + 1}/{max_attempts})")

    raise ValueError("Failed to generate a strong random key after maximum attempts.")

# Main flow
if __name__ == "__main__":
    print("Generating keys using quantum randomness...")

    # Generate a quantum-based random key (32 bytes)
    quantum_key = quantum_seeded_key()

    # The quantum key is now a 32-byte byte string
    print(f"\nGenerated Quantum Key (bytes): {quantum_key}")
    print(f"Generated Quantum Key (bytes in hex): {quantum_key.hex()}")

def kyber_ccakem_keygen():
        """
        Implements Kyber.CCAKEM.KeyGen algorithm as per the Kyber specification.

        Returns:
            - pk (bytes): The public key.
            - sk (bytes): The secret key.
        """
        # Step 1: Generate a random seed z (32 bytes)
    
        z = quantum_key

    # Use the OQS Kyber KEM
        kemalg = "Kyber512"

        with oqs.KeyEncapsulation(kemalg) as client:
            with oqs.KeyEncapsulation(kemalg) as server:
              pk = client.generate_keypair()
            sk_cpa = client.export_secret_key()

            h_pk = hashlib.sha3_256(pk).digest()

            sk = sk_cpa + pk + h_pk + z

            print("\nKey encapsulation details:")
            pprint(client.details)

        return pk, sk, sk_cpa
        
if __name__ == "__main__":
            public_key, secret_key = kyber_ccakem_keygen()
                
    
            print("\nClient Public Key:")
            print(public_key.hex())
            print("\nClient Secret Key:")
            print(secret_key.hex())

            # Server encapsulates its secret using the client's public key
            ciphertext, shared_secret_server = server.encap_secret(public_key_client)
            print("\nCiphertext (from server encapsulating using client's public key):")
            print(ciphertext.hex())

            # Client decapsulates the server's ciphertext to obtain the shared secret
            shared_secret_client = client.decap_secret(ciphertext)

            print(f"\nShared Secret from client: {shared_secret_client.hex()}")
            print(f"Shared Secret from server: {shared_secret_server.hex()}")

            # Check if the shared secrets coincide
            print("\nShared secrets coincide:", shared_secret_client == shared_secret_server)

            # Enhance shared secret by combining it with the quantum key
            enhanced_shared_secret = shared_secret_client[:16] + quantum_key
            print(f"\nEnhanced Shared Secret: {enhanced_shared_secret.hex()}")