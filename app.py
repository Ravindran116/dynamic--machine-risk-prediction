from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sqlite3


app = Flask(__name__)

# ==========================================
# DATABASE SETUP
# ==========================================

def init_db():

    conn = sqlite3.connect("machines.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_name TEXT NOT NULL,
            temperature REAL NOT NULL,
            pressure REAL NOT NULL,
            vibration TEXT NOT NULL,
            humidity REAL NOT NULL,
            risk TEXT
        )
    """)

    conn.commit()
    conn.close()


# Create database when application starts
init_db()


# Load the trained ML model
model = joblib.load("machine_risk_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# ML prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        temperature = float(data["Temperature"])
        pressure = float(data["Pressure"])
        vibration = data["Vibration"]
        humidity = float(data["Humidity"])

        # Create input data for the ML model
        input_data = pd.DataFrame([
            {
                "Temperature": temperature,
                "Pressure": pressure,
                "Vibration": vibration,
                "Humidity": humidity
            }
        ])

        # Ask the trained ML model for prediction
        prediction = model.predict(input_data)[0]
        print("ML INPUT:", input_data)
        print("ML PREDICTION:", prediction)

        return jsonify({
            "success": True,
            "risk": str(prediction),
            "source":"Python ML Model",
            "model": "Random Forest"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    # ==========================================
# SAVE MACHINE TO DATABASE
# ==========================================

@app.route("/machines", methods=["POST"])
def add_machine():

    try:

        data = request.get_json()

        machine_name = data["Machine Name"]
        temperature = float(data["Temperature"])
        pressure = float(data["Pressure"])
        vibration = data["Vibration"]
        humidity = float(data["Humidity"])
        risk = data.get("risk", "Unknown")

        conn = sqlite3.connect("machines.db")

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO machines
            (machine_name, temperature, pressure, vibration, humidity, risk)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            machine_name,
            temperature,
            pressure,
            vibration,
            humidity,
            risk
        ))

        conn.commit()

        machine_id = cursor.lastrowid

        conn.close()

        return jsonify({
            "success": True,
            "message": "Machine saved successfully",
            "id": machine_id
        })

    except Exception as e:
        print("DATABASE ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    
    # GET ALL MACHINES FROM DATABASE
@app.route("/machines", methods=["GET"])
def get_machines():

    try:
        conn = sqlite3.connect("machines.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM machines")

        machines = cursor.fetchall()

        conn.close()

        return jsonify({
            "success": True,
            "machines": [dict(machine) for machine in machines]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
# UPDATE MACHINE
@app.route("/machines/<int:machine_id>", methods=["PUT"])
def update_machine(machine_id):

    try:
        data = request.get_json()

        machine_name = data["Machine Name"]
        temperature = float(data["Temperature"])
        pressure = float(data["Pressure"])
        vibration = data["Vibration"]
        humidity = float(data["Humidity"])
        risk = data.get("risk", "Unknown")

        conn = sqlite3.connect("machines.db")
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE machines
            SET machine_name = ?,
                temperature = ?,
                pressure = ?,
                vibration = ?,
                humidity = ?,
                risk = ?
            WHERE id = ?
        """, (
            machine_name,
            temperature,
            pressure,
            vibration,
            humidity,
            risk,
            machine_id
        ))

        conn.commit()

        if cursor.rowcount == 0:
            conn.close()

            return jsonify({
                "success": False,
                "error": "Machine not found"
            }), 404

        conn.close()

        return jsonify({
            "success": True,
            "message": "Machine updated successfully"
        })

    except Exception as e:

        print("DATABASE UPDATE ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# DELETE MACHINE
@app.route("/machines/<int:machine_id>", methods=["DELETE"])
def delete_machine(machine_id):

    try:

        conn = sqlite3.connect("machines.db")
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM machines WHERE id = ?",
            (machine_id,)
        )

        conn.commit()

        if cursor.rowcount == 0:
            conn.close()

            return jsonify({
                "success": False,
                "error": "Machine not found"
            }), 404

        conn.close()

        return jsonify({
            "success": True,
            "message": "Machine deleted successfully"
        })

    except Exception as e:

        print("DATABASE DELETE ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)