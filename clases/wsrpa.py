import base64
import json
import aiohttp


class RPA:
    def __init__(self, path_ws_rpa):
        self.urlconf = {
            "pathWsRPA": path_ws_rpa
        }

    # MODELOS
    async def modelos_alt(self, nombre, descripcion):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ModelosALT"
        payload = {
            "Nombre": nombre,
            "Descripcion": descripcion
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en modelos_alt: {e}")
                raise

    async def modelos_act(self, id_modelo, nombre, descripcion, activo):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ModelosACT"
        payload = {
            "Id_Modelo": id_modelo,
            "Nombre": nombre,
            "Descripcion": descripcion,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en modelos_act: {e}")
                raise

    async def modelos_con(self, id_modelo=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ModelosCON"
        payload = {
            "Id_Modelo": id_modelo,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en modelos_con: {e}")
                raise

    async def modelos_del(self, id_modelo=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ModelosDEL"
        payload = {
            "Id_Modelo": id_modelo,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en modelos_del: {e}")
                raise

    # VERSIONES
    async def versiones_alt(self, nombre=None, modelo_id=None, id_drive=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/VersionesALT"
        payload = {
            "Nombre": nombre,
            "Modelo_Id": modelo_id,
            "Id_Drive": id_drive,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en versiones_alt: {e}")
                raise

    async def versiones_act(self, id_version=None, nombre=None, modelo_id=None, dataset_id=None, id_drivematriz=None, id_driveresultado=None, report=None,  activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/VersionesACT"
        payload = {
            "Id_Version": id_version,
            "Nombre": nombre,
            "Modelo_Id": modelo_id,
            "DataSet_Id": dataset_id,
            "Id_DriveMatriz": id_drivematriz,
            "Id_DriveResultado": id_driveresultado,
            "Report": report,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en versiones_act: {e}")
                raise

    async def versiones_con(self, id_version=None, modelo_id=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/VersionesCON"
        payload = {
            "Id_Version": id_version,
            "Modelo_Id": modelo_id
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"][0]
                    return False
            except Exception as e:
                print(f"Error en versiones_con: {e}")
                raise

    async def versiones_del(self, id_version=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/VersionesDEL"
        payload = {
            "Id_Version": id_version
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en versiones_del: {e}")
                raise

    # CATEGORIAS
    async def categorias_alt(self, nombre=None, modelo_id=None, nombre_carpeta=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/CategoriasALT"
        payload = {
            "Nombre": nombre,
            "Modelo_Id": modelo_id,
            "NombreCarpeta": nombre_carpeta,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en categorias_alt: {e}")
                raise

    async def categorias_act(self, id_categoria=None, nombre=None, modelo_id=None, nombre_carpeta=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/CategoriasACT"
        payload = {
            "Id_Categoria": id_categoria,
            "Nombre": nombre,
            "Modelo_Id": modelo_id,
            "NombreCarpeta": nombre_carpeta,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en categorias_act: {e}")
                raise

    async def categorias_con(self, id_categoria=None, nombre=None, modelo_id=None, nombre_carpeta=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/CategoriasCON"
        payload = {
            "Id_Categoria": id_categoria,
            "Nombre": nombre,
            "Modelo_Id": modelo_id,
            "NombreCarpeta": nombre_carpeta,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en categorias_con: {e}")
                raise

    async def categorias_del(self, id_categoria=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/CategoriasDEL"
        payload = {
            "Id_Categoria": id_categoria
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en categorias_del: {e}")
                raise

    # CONFIGURACIONES
    async def configuraciones_alt(self, nombre=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ConfiguracionesALT"
        payload = {
            "Nombre": nombre,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en configuraciones_alt: {e}")
                raise

    async def configuraciones_act(self, id_configuracion=None, nombre=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ConfiguracionesACT"
        payload = {
            "Id_Configuracion": id_configuracion,
            "Nombre": nombre,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en configuraciones_act: {e}")
                raise

    async def configuraciones_con(self, version_id=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ConfiguracionesBotCON"
        payload = {
            "Version_Id": version_id,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en configuraciones_con: {e}")
                raise

    async def configuraciones_del(self, id_configuracion=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/ConfiguracionesDEL"
        payload = {
            "Id_Configuracion": id_configuracion
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en configuraciones_del: {e}")
                raise

    # PENDIENTES
    async def pendientes_alt(self, id_drive_data_set=None, tipo_pendiente=None, estatus_id=None, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/PendientesALT"
        payload = {
            "Id_DriveDataSet": id_drive_data_set,
            "TipoPendiente": tipo_pendiente,
            "Estatus_Id": estatus_id,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en pendientes_alt: {e}")
                raise

    async def pendientes_act(self, id_pendiente=None,  estatus_id=None, activo=None, detalle=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/PendientesACT"
        payload = {
            "Id_Pendiente": id_pendiente,
            "Estatus_Id": estatus_id,
            "Detalle": detalle,
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en pendientes_act: {e}")
                raise

    async def pendientes_con(self, activo=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/PendientesBotCON"
        payload = {
            "Activo": activo
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    else:
                        return []
            except Exception as e:
                print(f"Error en pendientes_con: {e}")
                raise

    async def pendientes_del(self, id_pendiente=None):
        url = f"{self.urlconf['pathWsRPA']}/api/MlPredictor/PendientesDEL"
        payload = {
            "Id_Pendiente": id_pendiente
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()

                    if "table" in data and len(data["table"]) > 0:
                        return data["table"]
                    return False
            except Exception as e:
                print(f"Error en pendientes_del: {e}")
                raise
