import os
import json
import requests
from ast import literal_eval

# Ruta absoluta al archivo de configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_config_file = os.path.join(BASE_DIR, "config.json")

def get_credentials():
    """
    Carga credenciales y configuraciones desde un archivo local (Configuracion/config.json)
    y, si es posible, desde una API remota definida en 'wsRPA'.
    """
    diccionario = {
        'global_host': '127.0.0.1',
        'global_user': 'root',
        'global_password': '',
        'global_db': 'globaldb'
    }

    try:
        # 1️⃣ Leer configuración local
        if os.path.exists(_config_file):
            with open(_config_file, 'r', encoding='utf-8') as file:
                local_config = json.load(file)
                diccionario.update(local_config)
        else:
            print(f"[AVISO] Archivo {_config_file} no encontrado. Usando valores por defecto.")

        # 2️⃣ Leer parámetros para conexión remota
        wsRPA = diccionario.get("wsRPA")
        sistema = diccionario.get("Sistema")
        if wsRPA and sistema:
            url = f"{wsRPA}/Api/General/AdminConfiguracionIniPhytoniniCON"
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(url, headers=headers, json={}, timeout=90)
                response.raise_for_status()
                data = response.json()
                # Buscar configuración del sistema
                table = data.get('table', [])
                config = next((item for item in table if item.get('sistema') == sistema), None)
                if config and config.get("jsonconfig"):
                    remote_config = json.loads(config["jsonconfig"])
                    diccionario.update(remote_config)
                else:
                    print(f"[AVISO] No se encontró configuración remota para el sistema '{sistema}'.")
            except requests.RequestException as e:
                print(f"[ERROR HTTP] No se pudo obtener configuración remota: {e}")
        else:
            print("[AVISO] No se encontraron los parámetros 'wsRPA' o 'Sistema' en la configuración local.")
    except json.JSONDecodeError as e:
        print(f"[ERROR] El archivo {_config_file} contiene JSON inválido: {e}")
        # Intentar cargar usando literal_eval si el JSON está mal formateado
        try:
            with open(_config_file, 'r', encoding='utf-8') as file:
                diccionario.update(literal_eval(file.read()))
        except Exception as e2:
            print(f"[ERROR] No se pudo cargar la configuración con literal_eval: {e2}")

    except Exception as e:
        print(f"[ERROR INESPERADO] {e}")
    return diccionario
