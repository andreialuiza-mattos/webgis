import streamlit as st
#import leafmap.foliumap as leafmap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from datetime import datetime
import altair as alt
from io import BytesIO
import base64
import folium
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html as st_html

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =============================================================================
# Define título, ícone, layout largo e sidebar expandida por padrão.
st.set_page_config(
    page_title="WebGIS - Análise Geoespacial",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal da aplicação + linha divisória
st.title("🌍 WebGIS - Análise Geoespacial Avançada")
st.markdown("---")

# =============================================================================
# CONFIGURAÇÕES DE ESTILOS DE MAPA (ATUALIZADO)
# =============================================================================
# Dicionário central com todos os basemaps disponíveis.
# Cada entrada precisa ter:
#   - "tiles":      template de URL ou nome nativo do Folium/Leafmap
#   - "attribution": crédito obrigatório (Folium ≥ 0.15.0 exige attr em tiles custom)
#   - "subdomains" (opcional): lista de subdomínios {s} para paralelizar requisições
ESTILOS_MAPA = {
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    "Terrain": {
        "tiles": "Stamen Terrain",
        "attribution": 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    },
    "CartoDB": {
        "tiles": "CartoDB positron",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    },
    "Stamen Terrain": {
        "tiles": "Stamen Terrain", 
        "attribution": 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    },
    "Satélite (Esri)": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    },
    # ====== NOVAS OPÇÕES DE MAPA DE FUNDO ======
    # Google Maps (mapa rodoviário) - lyrs=m
    # Os subdomínios mt0..mt3 permitem carregamento paralelo e evitam throttling.
    "Google Maps": {
        "tiles": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attribution": "© Google Maps",
        "subdomains": ["mt0", "mt1", "mt2", "mt3"]
    },
    # Google Satélite (imagem) - lyrs=s
    # Se quiser híbrido (imagem + rótulos), troque para lyrs=y.
    "Google Satélite": {
        "tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attribution": "© Google Satellite",
        "subdomains": ["mt0", "mt1", "mt2", "mt3"]
    },
    # OpenTopoMap - alternativa 100% gratuita e sem chave, ideal para terreno
    # (curvas de nível, sombreamento, hidrografia).
    "OpenTopoMap (Gratuito)": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attribution": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        "subdomains": ["a", "b", "c"]
    }
}

# =============================================================================
# CLASSES
# =============================================================================

class GerenciadorCamadas:
    """Gerencia múltiplas camadas no mapa (máx. 3 por padrão)."""
    
    def __init__(self):
        self.camadas = {}
        self.max_camadas = 3
    
    def adicionar_camada(self, nome, gdf, cor=None, estilo=None, tamanho=6, opacidade=0.7):
        """Adiciona uma nova camada ao gerenciador. Retorna (sucesso, mensagem)."""
        # Bloqueia se o limite de camadas foi atingido
        if len(self.camadas) >= self.max_camadas:
            return False, "Máximo de 3 camadas atingido"
        # Bloqueia nomes duplicados
        if nome in self.camadas:
            return False, "Nome de camada já existe"
        
        # Armazena metadados da camada
        self.camadas[nome] = {
            'gdf': gdf,
            'cor': cor or 'blue',
            'estilo': estilo or 'circle',
            'tamanho': tamanho,
            'opacidade': opacidade,
            'visivel': True,
            'tipo_geometria': self._detectar_tipo_geometria(gdf)
        }
        return True, f"Camada '{nome}' adicionada com sucesso"
    
    def _detectar_tipo_geometria(self, gdf):
        """Detecta o tipo de geometria predominante (ponto, linha, polígono)."""
        try:
            if hasattr(gdf, 'geometry') and not gdf.geometry.empty:
                for geom in gdf.geometry:
                    if geom and not geom.is_empty:
                        if geom.geom_type in ['Point', 'MultiPoint']:
                            return 'ponto'
                        elif geom.geom_type in ['LineString', 'MultiLineString']:
                            return 'linha'
                        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                            return 'poligono'
        except Exception:
            pass
        return 'desconhecido'
    
    def remover_camada(self, nome):
        """Remove uma camada pelo nome."""
        if nome in self.camadas:
            del self.camadas[nome]
            return True, f"Camada '{nome}' removida"
        return False, "Camada não encontrada"
    
    def toggle_visibilidade(self, nome):
        """Liga/desliga a visibilidade de uma camada."""
        if nome in self.camadas:
            self.camadas[nome]['visivel'] = not self.camadas[nome]['visivel']
            return True
        return False
    
    def get_camadas_visiveis(self):
        """Retorna apenas camadas marcadas como visíveis."""
        return {nome: dados for nome, dados in self.camadas.items() if dados['visivel']}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def verificar_session_state():
    """Garante que o gerenciador de camadas exista em st.session_state."""
    if 'gerenciador_camadas' not in st.session_state:
        st.session_state.gerenciador_camadas = GerenciadorCamadas()
    return st.session_state.gerenciador_camadas

def carregar_shapefile(zip_file):
    """Carrega um shapefile a partir de um ZIP, garantindo CRS EPSG:4326."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            import zipfile
            # Extrai o ZIP em diretório temporário
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Procura o primeiro .shp dentro do ZIP
            shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
            if not shp_files:
                st.error("Nenhum arquivo .shp encontrado no ZIP")
                return None
            
            # Lê com GeoPandas
            shp_path = os.path.join(temp_dir, shp_files[0])
            gdf = gpd.read_file(shp_path)
            
            # Reprojecta para WGS84 se necessário
            if gdf.crs and gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
                
            return gdf
    except Exception as e:
        st.error(f"Erro ao carregar shapefile: {str(e)}")
        return None

def carregar_dados(uploaded_file):
    """Carrega CSV, Excel ou shapefile (ZIP) e tenta inferir colunas lat/lon."""
    try:
        # ---------- CSV ----------
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Detecção automática de colunas 'latitude'/'longitude'
            if all(col in df.columns for col in ['latitude', 'longitude']):
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326"
                )
                return gdf
            # Detecção automática de colunas 'lat'/'lon'
            elif all(col in df.columns for col in ['lat', 'lon']):
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df.lon, df.lat),
                    crs="EPSG:4326"
                )
                return gdf
            else:
                st.warning("CSV carregado, mas colunas de latitude/longitude não detectadas automaticamente")
                return df
        
        # ---------- Excel ----------
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            # Busca colunas cujo nome contenha 'lat' / 'lon' ou 'lng'
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
            
            if lat_cols and lon_cols:
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df[lon_cols[0]], df[lat_cols[0]]),
                    crs="EPSG:4326"
                )
                return gdf
            else:
                st.warning("Excel carregado, mas colunas de latitude/longitude não detectadas automaticamente")
                return df
        
        # ---------- Shapefile (ZIP) ----------
        elif uploaded_file.name.endswith('.zip'):
            return carregar_shapefile(uploaded_file)
        
        else:
            st.error("Formato de arquivo não suportado")
            return None
            
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

def detectar_colunas_geo(df):
    """Tenta adivinhar colunas geográficas/temáticas pelo nome."""
    colunas = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'lat' in col_lower or 'y' in col_lower:
            colunas['latitude'] = col
        elif 'lon' in col_lower or 'x' in col_lower or 'lng' in col_lower:
            colunas['longitude'] = col
        elif 'bairro' in col_lower or 'neighborhood' in col_lower:
            colunas['bairro'] = col
        elif 'tipo' in col_lower or 'type' in col_lower or 'categoria' in col_lower:
            colunas['tipo'] = col
        elif 'nome' in col_lower or 'name' in col_lower:
            colunas['nome'] = col
    return colunas

def verificar_e_converter_tipos_numericos(gdf, coluna_cor):
    """Converte a coluna de classificação para numérico, se possível."""
    try:
        if coluna_cor in gdf.columns:
            if not pd.api.types.is_numeric_dtype(gdf[coluna_cor]):
                gdf[coluna_cor] = pd.to_numeric(gdf[coluna_cor], errors='coerce')
            # Remove linhas cujo valor virou NaN
            gdf_limpo = gdf.dropna(subset=[coluna_cor])
            if len(gdf_limpo) == 0:
                st.warning(f"Nenhum valor numérico válido encontrado na coluna '{coluna_cor}' após conversão")
                return None
            return gdf_limpo
        return gdf
    except Exception as e:
        st.error(f"Erro ao converter tipos numéricos: {str(e)}")
        return gdf

def preparar_dataframe_para_exibicao(df):
    """Remove geometria e converte datetime para string, para exibir no Streamlit."""
    df_copy = df.copy()
    if isinstance(df_copy, gpd.GeoDataFrame) and 'geometry' in df_copy.columns:
        df_copy = df_copy.drop(columns=['geometry'])
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(str)
    return df_copy

# =============================================================================
# FUNÇÕES DE MAPA
# =============================================================================

def obter_coordenadas_geometria(geometry):
    """Extrai (lat, lon) de uma geometria de forma segura."""
    try:
        if geometry.is_empty:
            return None
        if hasattr(geometry, 'x') and hasattr(geometry, 'y'):
            # Ponto simples
            return [geometry.y, geometry.x]
        elif hasattr(geometry, 'centroid'):
            # Linha/polígono → usa centroide
            centroid = geometry.centroid
            if not centroid.is_empty:
                return [centroid.y, centroid.x]
        elif hasattr(geometry, 'bounds'):
            # Fallback: centro do bounding box
            bounds = geometry.bounds
            if bounds:
                center_y = (bounds[1] + bounds[3]) / 2
                center_x = (bounds[0] + bounds[2]) / 2
                return [center_y, center_x]
    except Exception:
        return None
    return None

def criar_popup(row, nome_camada, colunas_excluir=['geometry']):
    """Monta o HTML do popup de uma feição."""
    try:
        popup_text = f"<b>{nome_camada}</b><br>"
        for col, value in row.items():
            if col not in colunas_excluir and not col.startswith('_'):
                if pd.isna(value):
                    value_str = 'N/A'
                else:
                    try:
                        value_str = str(value)
                        # Trunca textos muito longos
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                    except:
                        value_str = 'N/A'
                popup_text += f"<b>{col}:</b> {value_str}<br>"
        return popup_text
    except Exception:
        return f"<b>{nome_camada}</b><br>Informações disponíveis"

def determinar_cor_classificacao(row, gdf, coluna_cor, cor_base):
    """Classifica a cor de uma feição com base no valor da coluna escolhida."""
    # Sem coluna de classificação → usa cor base da camada
    if not coluna_cor or coluna_cor == "Nenhuma" or coluna_cor not in gdf.columns:
        return cor_base
    try:
        valor = row[coluna_cor]
        if pd.isna(valor):
            return cor_base
        if pd.api.types.is_numeric_dtype(gdf[coluna_cor]):
            # Classificação numérica em 5 faixas
            valores_validos = gdf[coluna_cor].dropna()
            if len(valores_validos) == 0:
                return cor_base
            min_val = valores_validos.min()
            max_val = valores_validos.max()
            if min_val == max_val:
                return 'blue'
            normalized = (valor - min_val) / (max_val - min_val)
            if normalized < 0.2:
                return 'green'
            elif normalized < 0.4:
                return 'lightgreen'
            elif normalized < 0.6:
                return 'yellow'
            elif normalized < 0.8:
                return 'orange'
            else:
                return 'red'
        else:
            # Classificação categórica: mapeia cada categoria a uma cor fixa
            categorias = gdf[coluna_cor].dropna().unique()
            cores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                    'lightred', 'darkblue', 'darkgreen', 'cadetblue']
            if valor in categorias:
                idx = list(categorias).index(valor)
                return cores[idx % len(cores)]
            else:
                return cor_base
    except Exception:
        return cor_base

def criar_mapa_folium_direto(gerenciador_camadas, estilo_mapa='OpenStreetMap', coluna_cor=None):
    """Cria um mapa Folium puro (com controle total de camadas).
    
    NOTA: usa config_mapa["attribution"] para satisfazer a exigência do Folium ≥ 0.15.0
    de fornecer 'attr' quando tiles é uma URL customizada (Google Maps, Google Satélite,
    Esri, OpenTopoMap...).
    """
    camadas_visiveis = gerenciador_camadas.get_camadas_visiveis()
    
    # ---- Sem camadas visíveis: retorna mapa vazio já configurado ----
    if not camadas_visiveis:
        config_mapa = ESTILOS_MAPA.get(estilo_mapa, ESTILOS_MAPA["OpenStreetMap"])
        m = folium.Map(
            location=[-15.7939, -47.8828], 
            zoom_start=4, 
            tiles=config_mapa["tiles"],
            attr=config_mapa["attribution"]
        )
        return m
    
    # ---- Calcula centro e zoom a partir da primeira camada ----
    primeira_camada = list(camadas_visiveis.values())[0]
    gdf_base = primeira_camada['gdf']
    location = [-15.7939, -47.8828]  # Centro padrão (Brasília)
    zoom_start = 4
    try:
        if hasattr(gdf_base, 'geometry') and not gdf_base.geometry.empty:
            coordenadas_validas = []
            for geom in gdf_base.geometry:
                coords = obter_coordenadas_geometria(geom)
                if coords:
                    coordenadas_validas.append(coords)
            if coordenadas_validas:
                avg_lat = np.mean([c[0] for c in coordenadas_validas])
                avg_lon = np.mean([c[1] for c in coordenadas_validas])
                location = [avg_lat, avg_lon]
                zoom_start = 10
    except Exception:
        pass
    
    # ---- Cria o mapa base com o estilo escolhido ----
    config_mapa = ESTILOS_MAPA.get(estilo_mapa, ESTILOS_MAPA["OpenStreetMap"])
    m = folium.Map(
        location=location, 
        zoom_start=zoom_start, 
        tiles=config_mapa["tiles"],
        attr=config_mapa["attribution"]
    )
    
    # ---- Adiciona cada camada visível ----
    for nome_camada, dados_camada in camadas_visiveis.items():
        gdf = dados_camada['gdf']
        cor_base = dados_camada['cor']
        tamanho = dados_camada.get('tamanho', 6)
        opacidade = dados_camada.get('opacidade', 0.7)
        tipo_geometria = dados_camada.get('tipo_geometria', 'desconhecido')
        
        feature_group = folium.FeatureGroup(name=nome_camada, show=True)
        features_adicionadas = 0
        
        try:
            for idx, row in gdf.iterrows():
                try:
                    geometry = row.geometry
                    if geometry is None or geometry.is_empty:
                        continue
                    coords = obter_coordenadas_geometria(geometry)
                    if not coords:
                        continue
                    
                    # Popup + cor calculada por feição
                    popup = folium.Popup(criar_popup(row, nome_camada), max_width=300)
                    cor_final = determinar_cor_classificacao(row, gdf, coluna_cor, cor_base)
                    
                    # ---- Pontos: CircleMarker ----
                    if tipo_geometria == 'ponto':
                        folium.CircleMarker(
                            location=coords,
                            radius=tamanho,
                            popup=popup,
                            color=cor_final,
                            fill=True,
                            fillColor=cor_final,
                            fillOpacity=opacidade,
                            weight=1
                        ).add_to(feature_group)
                        features_adicionadas += 1
                    
                    # ---- Linhas e polígonos: GeoJSON ----
                    elif tipo_geometria in ['linha', 'poligono']:
                        try:
                            geojson_feature = {
                                'type': 'Feature',
                                'geometry': geometry.__geo_interface__,
                                'properties': {}
                            }
                            if tipo_geometria == 'poligono':
                                folium.GeoJson(
                                    geojson_feature,
                                    style_function=lambda x, cor=cor_final: {
                                        'fillColor': cor, 'color': cor,
                                        'weight': 2, 'fillOpacity': opacidade
                                    }
                                ).add_child(popup).add_to(feature_group)
                            else:
                                folium.GeoJson(
                                    geojson_feature,
                                    style_function=lambda x, cor=cor_final: {
                                        'color': cor, 'weight': 3, 'opacity': opacidade
                                    }
                                ).add_child(popup).add_to(feature_group)
                            features_adicionadas += 1
                        except Exception:
                            # Fallback: marcador no centroide
                            folium.CircleMarker(
                                location=coords, radius=tamanho, popup=popup,
                                color=cor_final, fill=True, fillColor=cor_final,
                                fillOpacity=opacidade, weight=1
                            ).add_to(feature_group)
                            features_adicionadas += 1
                    
                    # ---- Tipo desconhecido: marcador genérico ----
                    else:
                        folium.CircleMarker(
                            location=coords, radius=tamanho, popup=popup,
                            color=cor_final, fill=True, fillColor=cor_final,
                            fillOpacity=opacidade, weight=1
                        ).add_to(feature_group)
                        features_adicionadas += 1
                except Exception:
                    continue  # Ignora feições problemáticas individualmente
            
            if features_adicionadas > 0:
                feature_group.add_to(m)
                st.success(f"✅ {features_adicionadas} feições adicionadas na camada '{nome_camada}'")
            else:
                st.warning(f"⚠️ Nenhuma feição válida encontrada na camada '{nome_camada}'")
        except Exception as e:
            st.error(f"Erro ao processar camada '{nome_camada}': {str(e)}")
    
    # Controle de camadas só faz sentido com >1 camada
    if len(camadas_visiveis) > 1:
        folium.LayerControl().add_to(m)
    return m

def criar_mapa_com_leafmap(gerenciador_camadas, estilo_mapa='OpenStreetMap', coluna_cor=None):
    """Versão alternativa usando Leafmap (usa add_basemap, que já cuida de atribuições)."""
    camadas_visiveis = gerenciador_camadas.get_camadas_visiveis()
    
    if not camadas_visiveis:
        m = leafmap.Map(center=[-15.7939, -47.8828], zoom=4)
        m.add_basemap(estilo_mapa)
        return m
    
    # Centro a partir da primeira camada
    primeira_camada = list(camadas_visiveis.values())[0]
    gdf_base = primeira_camada['gdf']
    center = [-15.7939, -47.8828]
    zoom = 4
    try:
        if hasattr(gdf_base, 'geometry') and not gdf_base.geometry.empty:
            coordenadas_validas = []
            for geom in gdf_base.geometry:
                coords = obter_coordenadas_geometria(geom)
                if coords:
                    coordenadas_validas.append(coords)
            if coordenadas_validas:
                center = [
                    np.mean([c[0] for c in coordenadas_validas]),
                    np.mean([c[1] for c in coordenadas_validas])
                ]
                zoom = 10
    except Exception:
        pass
    
    m = leafmap.Map(center=center, zoom=zoom)
    m.add_basemap(estilo_mapa)
    
    # Adiciona cada camada
    for nome_camada, dados_camada in camadas_visiveis.items():
        gdf = dados_camada['gdf']
        cor_base = dados_camada['cor']
        tamanho = dados_camada.get('tamanho', 6)
        opacidade = dados_camada.get('opacidade', 0.7)
        tipo_geometria = dados_camada.get('tipo_geometria', 'desconhecido')
        
        try:
            if tipo_geometria == 'ponto':
                # Pontos: usa Folium FeatureGroup + CircleMarker (mais controle)
                feature_group = folium.FeatureGroup(name=nome_camada, show=True)
                features_adicionadas = 0
                for idx, row in gdf.iterrows():
                    try:
                        geometry = row.geometry
                        if geometry is None or geometry.is_empty:
                            continue
                        coords = obter_coordenadas_geometria(geometry)
                        if not coords:
                            continue
                        popup = folium.Popup(criar_popup(row, nome_camada), max_width=300)
                        cor_final = determinar_cor_classificacao(row, gdf, coluna_cor, cor_base)
                        folium.CircleMarker(
                            location=coords, radius=tamanho, popup=popup,
                            color=cor_final, fill=True, fillColor=cor_final,
                            fillOpacity=opacidade, weight=1
                        ).add_to(feature_group)
                        features_adicionadas += 1
                    except Exception:
                        continue
                if features_adicionadas > 0:
                    feature_group.add_to(m)
                    st.success(f"✅ {features_adicionadas} pontos adicionados na camada '{nome_camada}' (CircleMarkers)")
                else:
                    st.warning(f"⚠️ Nenhum ponto válido encontrado na camada '{nome_camada}'")
            else:
                # Linhas/polígonos: usa o método nativo do Leafmap
                style = {
                    'color': cor_base,
                    'fillColor': cor_base,
                    'fillOpacity': opacidade,
                    'weight': 2
                }
                m.add_gdf(gdf, layer_name=nome_camada, style=style, zoom_to_layer=False)
                st.success(f"✅ Camada '{nome_camada}' adicionada via Leafmap ({tipo_geometria})")
        except Exception as e:
            st.error(f"Erro ao adicionar camada '{nome_camada}' com Leafmap: {str(e)}")
    
    m.add_layer_control()
    return m

def exibir_mapa_no_streamlit(mapa, metodo_mapa):
    """Renderiza o mapa no Streamlit, com tratamento de erro."""
    try:
        if metodo_mapa == "Folium Direto":
            # Folium → extrai HTML e injeta via components.v1.html
            st_html(mapa._repr_html_(), height=600, scrolling=False)
            return True
        else:
            # Leafmap → usa to_streamlit nativo
            mapa.to_streamlit(height=600)
            return True
    except Exception as e:
        st.error(f"Erro ao exibir mapa: {str(e)}")
        return False

def salvar_mapa_html(mapa, nome_arquivo=None):
    """Serializa o mapa para HTML (retorna bytes + nome do arquivo)."""
    try:
        if nome_arquivo is None:
            data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"mapa_{data_hora}.html"
        elif not nome_arquivo.endswith('.html'):
            nome_arquivo += '.html'
        
        if hasattr(mapa, '_repr_html_'):
            # Folium
            html_content = mapa._repr_html_().encode('utf-8')
        else:
            # Leafmap: salva em temp e lê de volta
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                mapa.to_html(tmp_file.name)
                tmp_file_path = tmp_file.name
            with open(tmp_file_path, 'rb') as f:
                html_content = f.read()
            os.unlink(tmp_file_path)
        return html_content, nome_arquivo
    except Exception as e:
        st.error(f"Erro ao salvar mapa: {str(e)}")
        return None, None

# =============================================================================
# FUNÇÕES DE LEGENDA E CLASSIFICAÇÃO
# =============================================================================

def criar_legenda_classificacao(gdf, coluna_cor):
    """Constrói legenda visual para a coluna de classificação escolhida."""
    if coluna_cor not in gdf.columns:
        return None
    try:
        # Coluna numérica → escala contínua em 5 faixas
        if pd.api.types.is_numeric_dtype(gdf[coluna_cor]):
            valores = pd.to_numeric(gdf[coluna_cor], errors='coerce').dropna()
            if len(valores) == 0:
                st.warning(f"Nenhum valor numérico válido na coluna '{coluna_cor}'")
                return
            min_val = float(valores.min())
            max_val = float(valores.max())
            st.subheader("🎨 Legenda de Classificação - Valores")
            st.markdown(f"**Mínimo**: {min_val:.2f}")
            st.markdown(f"**Máximo**: {max_val:.2f}")
            st.markdown(f"**Média**: {valores.mean():.2f}")
            st.markdown("**Escala de Cores:**")
            cores = ['green', 'lightgreen', 'yellow', 'orange', 'red']
            if min_val == max_val:
                st.markdown(
                    f"<span style='color: green; font-weight: bold;'>■</span> {min_val:.1f}",
                    unsafe_allow_html=True
                )
            else:
                classes = np.linspace(min_val, max_val, 6)
                for i in range(len(cores)):
                    st.markdown(
                        f"<span style='color: {cores[i]}; font-weight: bold;'>■</span> "
                        f"{classes[i]:.1f} - {classes[i+1]:.1f}",
                        unsafe_allow_html=True
                    )
        # Coluna categórica → até 10 categorias com cores distintas
        elif gdf[coluna_cor].dtype == 'object':
            categorias = gdf[coluna_cor].value_counts().head(10)
            cores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                     'lightred', 'darkblue', 'darkgreen', 'cadetblue']
            st.subheader("🎨 Legenda de Classificação - Categorias")
            col_leg1, col_leg2 = st.columns(2)
            with col_leg1:
                for i, (categoria, count) in enumerate(list(categorias.items())[:5]):
                    cor = cores[i % len(cores)]
                    st.markdown(
                        f"<span style='color: {cor}; font-weight: bold;'>■</span> "
                        f"**{categoria}**: {count} registros",
                        unsafe_allow_html=True
                    )
            with col_leg2:
                for i, (categoria, count) in enumerate(list(categorias.items())[5:10]):
                    cor = cores[(i+5) % len(cores)]
                    st.markdown(
                        f"<span style='color: {cor}; font-weight: bold;'>■</span> "
                        f"**{categoria}**: {count} registros",
                        unsafe_allow_html=True
                    )
            if len(gdf[coluna_cor].unique()) > 10:
                st.info(f"Mostrando 10 de {len(gdf[coluna_cor].unique())} categorias")
        else:
            st.info(f"Tipo de dados não suportado para classificação: {gdf[coluna_cor].dtype}")
    except Exception as e:
        st.error(f"Erro ao criar legenda de classificação: {str(e)}")

# =============================================================================
# FUNÇÕES DE ANÁLISE DE DADOS
# =============================================================================

def criar_tabelas_dinamicas_totais(df, coluna_categoria, coluna_valor=None):
    """Gera tabelas de contagem/percentual/soma e estatísticas gerais."""
    df_display = preparar_dataframe_para_exibicao(df)
    tabelas = []
    
    # Contagem + percentual por categoria
    if coluna_categoria and coluna_categoria in df_display.columns:
        contagem_por_categoria = df_display[coluna_categoria].value_counts().reset_index()
        contagem_por_categoria.columns = [coluna_categoria, 'Total de Registros']
        contagem_por_categoria['Percentual (%)'] = (
            contagem_por_categoria['Total de Registros'] / len(df_display) * 100
        ).round(2)
        tabelas.append(("📊 Contagem por " + coluna_categoria, contagem_por_categoria))
    
    # Soma da coluna de valor (se houver)
    if coluna_valor and coluna_valor != "Nenhuma" and coluna_valor in df_display.columns:
        soma_por_categoria = df_display.groupby(coluna_categoria)[coluna_valor].agg('count').round(2)
        soma_por_categoria.columns = ['Quantidade']
        soma_por_categoria = soma_por_categoria.reset_index()
        tabelas.append((f"💰 Soma de {coluna_valor} por {coluna_categoria}", soma_por_categoria))
    
    # Estatísticas gerais
    estatisticas_gerais = pd.DataFrame({
        'Estatística': ['Total de Registros', 'Total de Categorias', 'Data de Carregamento'],
        'Valor': [
            len(df_display),
            df_display[coluna_categoria].nunique() if coluna_categoria in df_display.columns else 'N/A',
            datetime.now().strftime('%d/%m/%Y %H:%M')
        ]
    })
    tabelas.append(("📈 Estatísticas Gerais", estatisticas_gerais))
    return tabelas

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    # Inicializa o gerenciador de camadas na sessão
    gerenciador_camadas = verificar_session_state()
    
    # -------------------- SIDEBAR: UPLOAD --------------------
    st.sidebar.header("📁 Upload de Dados")
    uploaded_file = st.sidebar.file_uploader(
        "Escolha um arquivo para análise",
        type=['csv', 'xlsx', 'zip'],
        help="Suporte para CSV, Excel e Shapefile (ZIP)"
    )
    
    dados = None
    colunas_detectadas = {}
    
    if uploaded_file is not None:
        with st.spinner("Carregando dados..."):
            dados = carregar_dados(uploaded_file)
        
        if dados is not None:
            st.sidebar.success(f"✅ Arquivo carregado: {uploaded_file.name}")
            
            # Informações resumidas sobre o GeoDataFrame
            if isinstance(dados, gpd.GeoDataFrame):
                st.sidebar.info(f"🌍 {len(dados)} feições carregadas")
                if hasattr(dados, 'crs'):
                    st.sidebar.info(f"📊 CRS: {dados.crs}")
                geometrias_validas = sum(1 for geom in dados.geometry if geom and not geom.is_empty)
                st.sidebar.info(f"📍 Geometrias válidas: {geometrias_validas}/{len(dados)}")
            else:
                st.sidebar.info(f"📊 {len(dados)} registros carregados")
                st.sidebar.info(f"📋 {len(dados.columns)} colunas")
            
            colunas_detectadas = detectar_colunas_geo(dados)
            
            with st.sidebar.expander("🔍 Visualizar Dados"):
                dados_display = preparar_dataframe_para_exibicao(dados)
                st.dataframe(dados_display.head(3))
    
    # -------------------- CONFIGURAÇÕES DE ANÁLISE --------------------
    if dados is not None:
        st.sidebar.header("⚙️ Configurações de Análise")
        dados_display = preparar_dataframe_para_exibicao(dados)
        colunas_disponiveis = dados_display.columns.tolist()
        
        # ---- Se não for GeoDataFrame, pede colunas lat/lon ----
        if not isinstance(dados, gpd.GeoDataFrame):
            st.sidebar.subheader("🌐 Configuração Geográfica")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                lat_col = st.selectbox(
                    "Coluna Latitude",
                    options=colunas_disponiveis,
                    index=colunas_disponiveis.index(colunas_detectadas.get('latitude', colunas_disponiveis[0]))
                    if colunas_detectadas.get('latitude') in colunas_disponiveis else 0
                )
            with col2:
                lon_col = st.selectbox(
                    "Coluna Longitude",
                    options=colunas_disponiveis,
                    index=colunas_disponiveis.index(colunas_detectadas.get('longitude', colunas_disponiveis[1]))
                    if colunas_detectadas.get('longitude') in colunas_disponiveis else 1
                )
            
            try:
                dados = gpd.GeoDataFrame(
                    dados,
                    geometry=gpd.points_from_xy(dados[lon_col], dados[lat_col]),
                    crs="EPSG:4326"
                )
                st.sidebar.success("✅ Dados convertidos para GeoDataFrame")
            except Exception as e:
                st.error(f"Erro ao converter para GeoDataFrame: {str(e)}")
                return
        
        # ---- Colunas de categoria/valor ----
        st.sidebar.subheader("📊 Configurações de Visualização")
        coluna_bairro = st.sidebar.selectbox(
            "Coluna para Categoria (Bairro/Tipo)",
            options=["Nenhuma"] + colunas_disponiveis,
            index=0
        )
        colunas_numericas = dados_display.select_dtypes(include=[np.number]).columns.tolist()
        coluna_valor = st.sidebar.selectbox(
            "Coluna para Valores (Opcional)",
            options=["Nenhuma"] + colunas_numericas,
            index=0
        )
        
        # ---- Estilo do mapa (agora lista TODAS as novas opções) ----
        st.sidebar.subheader("🗺️ Configurações do Mapa")
        estilo_mapa = st.sidebar.selectbox(
            "Estilo do Mapa",
            list(ESTILOS_MAPA.keys())
            # ↑ Aqui aparecerão: OpenStreetMap, Terrain, CartoDB, Stamen Terrain,
            #   Satélite (Esri), Google Maps, Google Satélite, OpenTopoMap (Gratuito)
        )
        
        # ---- Coluna de classificação por cores ----
        colunas_todas = dados.columns.tolist()
        coluna_cor = st.sidebar.selectbox(
            "Coluna para Classificação por Cores",
            options=["Nenhuma"] + colunas_todas,
            index=0,
            help="Selecione uma coluna para classificar os dados por cores no mapa"
        )

        # ---- Método de renderização do mapa ----
        metodo_mapa = st.sidebar.radio(
            "Método de Renderização:",
            ["Folium Direto"],
            help="Folium Direto: mais controle, Leafmap: mais simples"
        )

        # Info extra sobre a coluna de classificação
        if coluna_cor != "Nenhuma" and coluna_cor in dados.columns:
            if pd.api.types.is_numeric_dtype(dados[coluna_cor]):
                st.sidebar.info(f"📊 **{coluna_cor}**: Coluna numérica")
            elif dados[coluna_cor].dtype == 'object':
                st.sidebar.info(f"📝 **{coluna_cor}**: Coluna categórica ({len(dados[coluna_cor].unique())} categorias)")
            else:
                st.sidebar.info(f"🔍 **{coluna_cor}**: Tipo {dados[coluna_cor].dtype}")
        
        # -------------------- GERENCIAMENTO DE CAMADAS --------------------
        st.sidebar.subheader("🔄 Gerenciamento de Camadas")
        
        # Detecta tipo predominante de geometria
        tipo_geometria = "desconhecido"
        if hasattr(dados, 'geometry') and not dados.geometry.empty:
            tipos = dados.geometry.type.unique()
            if 'Point' in tipos or 'MultiPoint' in tipos:
                tipo_geometria = "pontos"
            elif 'LineString' in tipos or 'MultiLineString' in tipos:
                tipo_geometria = "linhas"
            elif 'Polygon' in tipos or 'MultiPolygon' in tipos:
                tipo_geometria = "polígonos"
        st.sidebar.info(f"📐 Tipo de geometria: {tipo_geometria}")
        
        # Configurações específicas para pontos
        usa_circlemarker = False
        tamanho_marker = 6
        opacidade_marker = 0.7
        if tipo_geometria == "pontos":
            st.sidebar.markdown("**🎯 Configurações de Markers:**")
            usa_circlemarker = st.sidebar.checkbox(
                "Usar CircleMarkers", value=True,
                help="Usar marcadores circulares interativos para pontos"
            )
            if usa_circlemarker:
                col_tam, col_op = st.sidebar.columns(2)
                with col_tam:
                    tamanho_marker = st.slider(
                        "Tamanho do marker", min_value=3, max_value=15, value=6,
                        help="Raio do CircleMarker em pixels"
                    )
                with col_op:
                    opacidade_marker = st.slider(
                        "Opacidade", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                        help="Opacidade do preenchimento"
                    )
        
        # Nome e cor da camada
        num_camadas = len(gerenciador_camadas.camadas)
        nome_camada = st.sidebar.text_input(
            "Nome da Camada:",
            value=f"Camada_{num_camadas + 1}",
            help="Digite um nome único para esta camada"
        )
        cor_camada = st.sidebar.color_picker("Cor da Camada:", value="#1f77b4")
        
        # Botões de ação
        col_add, col_view, col_clear = st.sidebar.columns(3)
        with col_add:
            if st.button("➕ Adicionar", use_container_width=True):
                estilo = "circlemarker" if (tipo_geometria == "pontos" and usa_circlemarker) else "padrao"
                sucesso, mensagem = gerenciador_camadas.adicionar_camada(
                    nome_camada, dados, cor_camada, estilo=estilo,
                    tamanho=tamanho_marker, opacidade=opacidade_marker
                )
                if sucesso:
                    st.sidebar.success(mensagem)
                    st.rerun()
                else:
                    st.sidebar.error(mensagem)
        with col_view:
            if st.button("🔄 Atualizar", use_container_width=True):
                st.sidebar.info(f"Camadas: {num_camadas}/3")
        with col_clear:
            if st.button("🗑️ Limpar", use_container_width=True):
                gerenciador_camadas.camadas = {}
                st.rerun()
        
        # Lista de camadas ativas
        if gerenciador_camadas.camadas:
            st.sidebar.markdown("**Camadas Ativas:**")
            for nome_camada, dados_camada in gerenciador_camadas.camadas.items():
                col_nome, col_vis, col_del = st.sidebar.columns([3, 1, 1])
                with col_nome:
                    estilo = "🔵" if dados_camada.get('estilo') == 'circlemarker' else "⬤"
                    st.write(f"{estilo} {nome_camada}")
                with col_vis:
                    icone = "👁️" if dados_camada['visivel'] else "🙈"
                    if st.button(icone, key=f"vis_{nome_camada}"):
                        gerenciador_camadas.toggle_visibilidade(nome_camada)
                        st.rerun()
                with col_del:
                    if st.button("❌", key=f"del_{nome_camada}"):
                        gerenciador_camadas.remover_camada(nome_camada)
                        st.rerun()
        
        # -------------------- ABAS PRINCIPAIS --------------------
        tab1, tab3, tab4 = st.tabs(["🗺️ Mapa Interativo", "📋 Dados", "📈 Estatísticas"])
        
        # ================= ABA 1: MAPA =================
        with tab1:
            st.header("🗺️ Mapa Interativo - Múltiplas Camadas")
            
            # Métricas rápidas
            total_camadas = len(gerenciador_camadas.camadas)
            camadas_visiveis = len(gerenciador_camadas.get_camadas_visiveis())
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total de Camadas", total_camadas)
            with col_info2:
                st.metric("Camadas Visíveis", camadas_visiveis)
            with col_info3:
                st.metric("Limite", f"{total_camadas}/3")
            
            if coluna_cor != "Nenhuma":
                st.info(f"🎨 Classificando por: **{coluna_cor}**")
            st.info(f"🔧 Método: **{metodo_mapa}**")
            # Mostra qual basemap está ativo (útil para confirmar as novas opções)
            st.info(f"🗺️ Mapa de fundo: **{estilo_mapa}**")
            
            # Cria e exibe o mapa
            with st.spinner("Gerando mapa com múltiplas camadas..."):
                try:
                    if metodo_mapa == "Folium Direto":
                        mapa = criar_mapa_folium_direto(gerenciador_camadas, estilo_mapa, coluna_cor)
                    else:
                        mapa = criar_mapa_com_leafmap(gerenciador_camadas, estilo_mapa, coluna_cor)
                    
                    if mapa:
                        st.success(f"✅ Mapa gerado com {camadas_visiveis} camada(s) visível(eis)")
                        exibiu = exibir_mapa_no_streamlit(mapa, metodo_mapa)
                        if not exibiu:
                            st.warning("⚠️ Não foi possível exibir o mapa interativo. Tente o outro método de renderização.")
                    else:
                        st.error("❌ Falha ao criar o mapa")
                except Exception as e:
                    st.error(f"Erro ao criar mapa: {str(e)}")
                    import traceback
                    st.error(f"Detalhes do erro: {traceback.format_exc()}")
            
            # Download do mapa
            st.markdown("---")
            st.subheader("💾 Download do Mapa")
            col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1])
            with col_dl1:
                nome_arquivo = st.text_input(
                    "Nome do arquivo HTML:",
                    value=f"mapa_multicamadas_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    help="Digite o nome do arquivo (sem extensão .html)"
                )
            with col_dl2:
                st.markdown("")
                if st.button("💾 Salvar Mapa", use_container_width=True):
                    with st.spinner("Salvando mapa..."):
                        try:
                            html_content, filename = salvar_mapa_html(mapa, nome_arquivo)
                            if html_content is not None:
                                st.success(f"Mapa salvo como {filename}")
                                st.session_state.html_content = html_content
                                st.session_state.filename = filename
                        except Exception as e:
                            st.error(f"Erro ao salvar mapa: {str(e)}")
            with col_dl3:
                st.markdown("")
                if 'html_content' in st.session_state and 'filename' in st.session_state:
                    st.download_button(
                        label="⬇️ Download",
                        data=st.session_state.html_content,
                        file_name=st.session_state.filename,
                        mime="text/html",
                        use_container_width=True
                    )
            
            # Legenda de classificação
            if coluna_cor != "Nenhuma" and coluna_cor in dados.columns:
                st.markdown("---")
                criar_legenda_classificacao(dados, coluna_cor)
            
            # Legenda das camadas
            if gerenciador_camadas.camadas:
                st.markdown("---")
                st.subheader("🗂️ Legenda das Camadas")
                col_leg1, col_leg2 = st.columns(2)
                camadas_lista = list(gerenciador_camadas.camadas.items())
                with col_leg1:
                    for nome_c, dados_c in camadas_lista[:len(camadas_lista)//2]:
                        cor = dados_c['cor']
                        visivel = "👁️ Visível" if dados_c['visivel'] else "🙈 Oculto"
                        count = len(dados_c['gdf'])
                        tipo = dados_c.get('tipo_geometria', 'desconhecido')
                        estilo = "🔵 CircleMarker" if dados_c.get('estilo') == 'circlemarker' else "⬤ Padrão"
                        st.markdown(
                            f"<span style='color: {cor}; font-weight: bold;'>■</span> "
                            f"**{nome_c}** - {count} feições<br>"
                            f"<small>{tipo} • {estilo} • {visivel}</small>",
                            unsafe_allow_html=True
                        )
                with col_leg2:
                    for nome_c, dados_c in camadas_lista[len(camadas_lista)//2:]:
                        cor = dados_c['cor']
                        visivel = "👁️ Visível" if dados_c['visivel'] else "🙈 Oculto"
                        count = len(dados_c['gdf'])
                        tipo = dados_c.get('tipo_geometria', 'desconhecido')
                        estilo = "🔵 CircleMarker" if dados_c.get('estilo') == 'circlemarker' else "⬤ Padrão"
                        st.markdown(
                            f"<span style='color: {cor}; font-weight: bold;'>■</span> "
                            f"**{nome_c}** - {count} feições<br>"
                            f"<small>{tipo} • {estilo} • {visivel}</small>",
                            unsafe_allow_html=True
                        )
        
        # ================= ABA 3: DADOS =================
        with tab3:
            st.header("📋 Dados Completos")
            st.info(f"Visualizando: {nome_camada if 'nome_camada' in locals() else 'Camada Principal'}")
            st.subheader("🔍 Filtros")
            col_f1, col_f2 = st.columns(2)
            dados_filtrados = preparar_dataframe_para_exibicao(dados)
            
            with col_f1:
                if coluna_bairro != "Nenhuma":
                    categorias = dados_filtrados[coluna_bairro].unique()
                    categorias_selecionadas = st.multiselect(
                        f"Filtrar por {coluna_bairro}",
                        options=categorias, default=categorias
                    )
                    if categorias_selecionadas:
                        dados_filtrados = dados_filtrados[
                            dados_filtrados[coluna_bairro].isin(categorias_selecionadas)
                        ]
            with col_f2:
                if coluna_valor != "Nenhuma" and coluna_valor in dados_filtrados.columns:
                    min_val = float(dados_filtrados[coluna_valor].min())
                    max_val = float(dados_filtrados[coluna_valor].max())
                    valor_range = st.slider(
                        f"Filtrar por {coluna_valor}",
                        min_val, max_val, (min_val, max_val)
                    )
                    dados_filtrados = dados_filtrados[
                        (dados_filtrados[coluna_valor] >= valor_range[0]) &
                        (dados_filtrados[coluna_valor] <= valor_range[1])
                    ]
            st.dataframe(dados_filtrados, height=400)
        
        # ================= ABA 4: ESTATÍSTICAS =================
        with tab4:
            st.header("📈 Estatísticas - Tabelas Dinâmicas")
            if coluna_bairro != "Nenhuma":
                tabelas = criar_tabelas_dinamicas_totais(dados, coluna_bairro, coluna_valor)
                for titulo, tabela in tabelas:
                    st.subheader(titulo)
                    st.dataframe(tabela)
                    st.markdown("---")
                
                # Estatísticas por camada
                if gerenciador_camadas.camadas:
                    st.subheader("📊 Estatísticas das Camadas")
                    stats_data = []
                    for nome_c, dados_c in gerenciador_camadas.camadas.items():
                        stats_data.append({
                            'Camada': nome_c,
                            'Feições': len(dados_c['gdf']),
                            'Visível': 'Sim' if dados_c['visivel'] else 'Não',
                            'Tipo': dados_c.get('tipo_geometria', 'desconhecido'),
                            'Cor': dados_c['cor']
                        })
                    st.dataframe(pd.DataFrame(stats_data))
            else:
                st.warning("Selecione uma coluna de categoria na sidebar para ver as estatísticas")
    
    else:
        # -------------------- TELA INICIAL (SEM ARQUIVO) --------------------
        st.markdown("""
        ## 🌟 Bem-vindo ao WebGIS Avançado!
        
        **Carregue seus dados geoespaciais para começar a análise:**
        
        ### 📁 Formatos Suportados:
        - **CSV**: Com colunas de latitude e longitude
        - **Excel**: Planilhas com dados geoespaciais  
        - **Shapefile**: Arquivo ZIP contendo .shp, .shx, .dbf, etc.
        
        ### 🗺️ Mapas de Fundo Disponíveis:
        - OpenStreetMap, Terrain, CartoDB, Stamen Terrain, Satélite (Esri)
        - **Google Maps** e **Google Satélite**
        - **OpenTopoMap** (totalmente gratuito)
        
        ### 🚀 Como Usar:
        1. **Faça upload** do arquivo na sidebar
        2. **Configure** as colunas geográficas (se necessário)
        3. **Escolha o mapa de fundo** na seção "Configurações do Mapa"
        4. **Adicione camadas** para comparar dados
        5. **Classifique por cores** para visualizar padrões
        6. **Explore** os dados nas diferentes abas
        """)
        
        # Exemplo de estrutura de dados
        st.subheader("📝 Exemplo de Estrutura de Dados")
        exemplo_data = pd.DataFrame({
            'latitude': [-23.5505, -22.9068, -23.5489, -19.9167, -15.7939],
            'longitude': [-46.6333, -43.1729, -46.6388, -43.9333, -47.8828],
            'bairro': ['Centro', 'Copacabana', 'Vila Madalena', 'Savassi', 'Asa Sul'],
            'tipo': ['Comercial', 'Residencial', 'Comercial', 'Residencial', 'Governamental'],
            'valor': [1000000, 500000, 800000, 600000, 1200000]
        })
        st.dataframe(exemplo_data)

if __name__ == "__main__":
    main()
