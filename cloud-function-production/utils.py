import numpy as np
import pandas as pd
import datetime

rng = np.random.default_rng(42)


def generate_transactions(n=10000, fraud_rate=0.02):
    transactions = []

    types = [
        "CASH_IN",
        "CASH_OUT",
        "PAYMENT",
        "TRANSFER",
        "DEBIT"
    ]

    probabilities = [0.22, 0.35, 0.32, 0.09, 0.02]

    for _ in range(n):
        is_fraud = int(rng.random() < fraud_rate)

        if is_fraud:
            tx_type = rng.choice(["TRANSFER", "CASH_OUT"])
            amount = round(rng.uniform(1000, 500000), 2)
            old_org = round(rng.uniform(amount, amount + 100000), 2)
        else:
            tx_type = rng.choice(types, p=probabilities)
            amount = round(rng.lognormal(10, 1.2), 2)
            old_org = round(rng.uniform(0, 1000000), 2)

        old_dest = round(rng.uniform(0, 1000000), 2)

        if tx_type == "CASH_IN":
            new_org = old_org + amount
            new_dest = old_dest
        else:
            new_org = max(0, old_org - amount)

            if tx_type in ["TRANSFER", "CASH_OUT"]:
                new_dest = old_dest + amount
            else:
                new_dest = old_dest

        # Règle simplifiée inspirée de PaySim
        is_flagged = int(
            tx_type == "TRANSFER" and amount > 200000
        )

        transactions.append({
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": round(new_org, 2),
            "oldbalanceDest": old_dest,
            "newbalanceDest": round(new_dest, 2),
            "isFlaggedFraud": is_flagged,
            "isFraud": is_fraud
        })

    df = pd.DataFrame(transactions)

    return df


if __name__ == "__main__":
    df = generate_transactions(
        n=10,
        fraud_rate=0.02
    )
    df.to_csv("simulation.csv", index=False)