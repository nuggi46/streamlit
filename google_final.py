
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
#import pyarrow.parquet as pq 
#from streamlit import radio, sidebar, markdown, title, image, checkbox, selectbox, container, columns, multiselect, button, table, image


@st.cache_data
def get_restaurante(fn):
    return pd.read_parquet(ruta1)
ruta1 = "https://storage.googleapis.com/yelp-and-maps-data-processed/df_resto_user_final.parquet"
df_restaurante = get_restaurante(ruta1)


df_florida = df_restaurante[df_restaurante['ubicacion'] == 'Florida']
df_pennsylvania = df_restaurante[df_restaurante['ubicacion'] == 'Pennsylvania']
df_user=df_restaurante['user_id'].unique()
df_user2=df_user[:20]

def get_user(fn2):
    return pd.read_parquet(ruta2)
ruta2 = "https://storage.googleapis.com/yelp-and-maps-data-processed/highRest_dummies.parquet"
highdf = get_user(ruta2)

# Sidebar con opciones
st.sidebar.image("https://github.com/mreliflores/PF-Henry/blob/main/Sprint%233/streamlit/innovaLogo.jpeg?raw=true", width=300)
sidebar_option = st.sidebar.radio("Selecciona una opción", ["Inicio", "Decido yo", "Decide el mejor","Dashboard"])



#Fondo del sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

########### Se escribe aca el codigo de recomendacion y luego se visualiza mas adelante
#### Código de modelo de recomendación Parte 1 - Cascada de modelos en función de la variable

### Matriz
# Tomamos las primeras 50000 filas
df_sample = df_restaurante.head(1000)

# Creamos la matriz de usuario-restaurante utilizando pivot_table
user_resto_matrix = df_sample.pivot_table(index='user_id', columns='gmap_id', values='sentimiento_etiqueta', fill_value=0)

# Calculamos la similitud del coseno entre usuarios
user_similarity = cosine_similarity(user_resto_matrix)

# Convertimos la matriz de similitud a un df para mayor claridad
user_similarity_df = pd.DataFrame(user_similarity, index=user_resto_matrix.index, columns=user_resto_matrix.index)


def recommend_restaurants(df_restaurante, highdf, category=None, trend=None, location=None, city=None):
    # Copiamos el df original
    df_filtered = df_restaurante.copy()

    # Aplicamos filtros
    if category:
        df_filtered = df_filtered[df_filtered[category] == 1]

    if trend is not None:
        sentiment_filter = 1 if trend.lower() == 'positive' else 0
        df_filtered = df_filtered[df_filtered['sentimiento_etiqueta'] == sentiment_filter]

    if location:
        df_filtered = df_filtered[df_filtered['ubicacion'] == location]

    if city:
        df_filtered = df_filtered[df_filtered['city'] == city]

    # Verificamos la coincidencia de ubicación y ciudad
    if location and city and df_filtered.empty:
        return "No hay recomendaciones para los datos ingresados, por favor verifique e intente nuevamente."

    # Ordenamos por sentimiento
    df_filtered = df_filtered.sort_values(by='sentimiento_etiqueta', ascending=False)

    # Realizamos un seguimiento de los restaurantes recomendados
    recommended_restaurants = set()

    # Almacenamos las recomendaciones
    recommendations_list = []

    # Iteramos sobre el df filtrado
    for _, row in df_filtered.iterrows():
        restaurant_id = row['gmap_id']

        # Verificamos si el restaurante ya ha sido recomendado
        if restaurant_id not in recommended_restaurants:
            # Agregamos el restaurante al conjunto de recomendados
            recommended_restaurants.add(restaurant_id)

            # Buscamos el highlight usando gmap_id
            highlight_row = highdf.loc[highdf['gmap_id'] == restaurant_id]

            # Lista para almacenar los aspectos destacados
            highlights_list = [column for column in highdf.columns[1:] if highlight_row[column].values == 1]

            # Agregamos la fila del restaurante a la lista de recomendaciones
            row_data = row[['name', 'avg_rating', 'address2', 'city', 'ubicacion']].tolist()
            row_data.append('No Highlight' if not highlights_list else ', '.join(highlights_list))
            recommendations_list.append(row_data)

            # Detenemos el bucle si hemos alcanzado el límite de 3 recomendaciones
            if len(recommendations_list) == 3:
                break   

    # Creamos un df a partir de la lista de recomendaciones
    result = pd.DataFrame(recommendations_list, columns=['name', 'avg_rating', 'address2', 'city', 'ubicacion', 'highlights'])

    # Verificamos si hay al menos una recomendación
    if result.empty:
        return "No hay recomendaciones para los datos ingresados, por favor verifique e intente nuevamente."
    
    return result

################# Codigo modelo recomendación Parte 2 - recomienda en base a usuarios

def recommend_restaurants_for_user(user_id, user_resto_matrix, user_similarity_df, df_restaurante, top_n=3):
    """
    Genera recomendaciones de restaurantes para un usuario específico basado en la similitud de coseno.

    Parámetros:
    - user_id: ID del usuario para el cual deseas generar recomendaciones.
    - user_resto_matrix: Matriz de usuario-restaurante.
    - user_similarity_df: DF de similitud del coseno entre usuarios.
    - df_resto_user: DataFrame original de restaurantes y usuarios.
    - top_n: Número de recomendaciones a generar (por defecto, 3).

    Retorna:
    - DF con las recomendaciones, ordenadas por puntuación descendente, y el DataFrame filtrado.
    """
    
    # Verificarmos si el ID del usuario está presente en el índice de la matriz
    user_id = int(user_id)  # Convertir la cadena a entero
    if user_id not in user_resto_matrix.index:
        raise ValueError(f"El ID de usuario {user_id} no está presente en la matriz.")

    # Obtenemos las calificaciones del usuario
    user_ratings = user_resto_matrix.loc[user_id, :]

    # Calculamos la puntuación ponderada por similitud del coseno para cada restaurante
    weighted_scores = user_similarity_df.loc[user_id, :] @ user_resto_matrix.values

    # Creamos un df con las puntuaciones ponderadas
    recommendations_df = pd.DataFrame(data={'weighted_score': weighted_scores}, index=user_resto_matrix.columns)

    # Filtramos los restaurantes que el usuario ya ha calificado
    unrated_restaurants = recommendations_df[user_ratings == 0]

    # Ordenamos
    top_recommendations = unrated_restaurants.sort_values(by='weighted_score', ascending=False).head(top_n)

    # Filtramos el DataFrame original con las recomendaciones
    filtered_df = df_restaurante[df_restaurante['gmap_id'].isin(top_recommendations.index)]

    # Imprimimos el ID de usuario
    print(f"ID de usuario: {user_id}")

    # Eliminamos duplicados y luego imprimimos las recomendaciones excluyendo la columna 'gmap_id'
    unique_filtered_df = filtered_df[['name', 'address']].drop_duplicates()
    #print("Recomendaciones:")
    #print(unique_filtered_df.reset_index(drop=True))
    final=unique_filtered_df.reset_index(drop=True)
    #return top_recommendations, filtered_df
    return final

#################

#sección recomendación atributo

def recommend_atributo():
    #options = st.multiselect('Selecciona a continuación que debemos   \
    #tener en cuenta para darte la mejor opción del dia de hoy'\
    #    , ['Tipo de comida', 'Ubicación','Tendencia' ,'Precio'])
    
    container = st.container()
    all = st.checkbox("Selecciona todos")
 
    if all:
        selected_options = container.multiselect("Selecciona a continuación que debemos   \
    tener en cuenta para darte la mejor opción del dia de hoy:",
             ['Categoria','Estado','Ciudad','Tendencia'],['Categoria','Estado','Ciudad','Tendencia'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        categoria = col1.selectbox('Tipo de comida',[ 'Mexican restaurant', 'American restaurant', \
        'Breakfast restaurant','Chinese restaurant', 'Bar & grill', 'Italian restaurant', \
        'Hamburger restaurant', 'Seafood restaurant', 'Asian restaurant'])
        
        ubicacion = col2.selectbox("Estado", ['Florida','Pennsylvania'])
        if ubicacion == 'Florida':
            ciudad = col3.selectbox("Ciudad", df_florida["city"].unique())
        else:
            ciudad = col3.selectbox("Ciudad", df_pennsylvania["city"].unique())
            
        rating = col4.selectbox("Tendencia",['positive','negative']) 
        
        if st.button("Generar Recomendaciones"):
            recomendaciones = recommend_restaurants(df_restaurante, highdf, category=categoria, trend=rating,\
                location=ubicacion, city=ciudad)
            # Mostrar la tabla
            st.table(recomendaciones)
        
    else:
        selected_options =  container.multiselect("Selecciona a continuación que debemos   \
    tener en cuenta para darte la mejor opción del dia de hoy:",
            ['Categoria','Estado','Ciudad','Tendencia'])
        
        col1, col2, col3, col4 = st.columns(4)
        if 'Categoria' in selected_options:
            categoria = col1.selectbox('Tipo de comida',[ 'Mexican restaurant', 'American restaurant', \
        'Breakfast restaurant','Chinese restaurant', 'Bar & grill', 'Italian restaurant', \
        'Hamburger restaurant', 'Seafood restaurant', 'Asian restaurant'])
            
        if 'Estado' in selected_options:
            ubicacion = col2.selectbox("Estado", ['Florida','Pennsylvania'])
        
        if 'Ciudad' in selected_options:
            if ubicacion == 'Florida':
                ciudad = col3.selectbox("Ciudad", df_florida["city"].unique())
            else:
                ciudad = col3.selectbox("Ciudad", df_pennsylvania["city"].unique())
        
        if 'Tendencia' in selected_options:
            rating = col4.selectbox("Tendencia",['positive','negative']) 
        
            # Botón para generar recomendaciones
        if st.button("Generar Recomendaciones"):
            
            if 'Categoria' in selected_options and len(selected_options)==1:
                # Corregir el cálculo de recomendaciones
                recomendaciones = recommend_restaurants(df_restaurante, highdf, category=categoria, trend=None, location=None, city=None)
                # Mostrar la tabla
                st.table(recomendaciones)
            
            if 'Estado' in selected_options and len(selected_options)==1:
                # Corregir el cálculo de recomendaciones
                recomendaciones = recommend_restaurants(df_restaurante, highdf, category=None, trend=None, location=ubicacion, city=None)
                # Mostrar la tabla
                st.table(recomendaciones)            
                   
            if 'Categoria' and 'Estado' in selected_options and len(selected_options)==2:
                # Corregir el cálculo de recomendaciones
                recomendaciones = recommend_restaurants(df_restaurante, highdf, category=categoria, trend=None, location=ubicacion, city=None)
                # Mostrar la tabla
                st.table(recomendaciones)
                
                
            if 'Categoria' and 'Estado'  and 'Ciudad' in selected_options and len(selected_options)==3:
                # Corregir el cálculo de recomendaciones
                recomendaciones = recommend_restaurants(df_restaurante, highdf, category=categoria, trend=None, location=ubicacion, city=ciudad)
                # Mostrar la tabla
                st.table(recomendaciones)  
                
            if 'Categoria' and 'Estado' and 'Ciudad' and 'Tendencia' in selected_options and len(selected_options)==4: 
                # Corregir el cálculo de recomendaciones
                recomendaciones = recommend_restaurants(df_restaurante, highdf, category=categoria, trend=rating, location=ubicacion, city=ciudad)
                # Mostrar la tabla
                st.table(recomendaciones)                 
                
    
#sección recomendación influencer
def recommend_foodie():
# Crear un menú desplegable con la lista de películas para el sistema de recomendación
    influencer_referencia = st.selectbox('Seleccion un influencer de referencia', df_user2)  
    # Botón para generar recomendaciones
    if st.button("Generar Recomendaciones"):
            
        if  influencer_referencia:
            # Corregir el cálculo de recomendaciones
            recommendations_1 = recommend_restaurants_for_user(influencer_referencia, user_resto_matrix, user_similarity_df, df_restaurante)
            # Mostrar la tabla
            st.table(recommendations_1)

######################################### Esto lo hizo Elí ################
def dashboard():
    
    st.title("Espacio para empresas")

    toBusiness = '<p style="font-family:Source Sans Pro; color:Black; font-size: 16px;">Aquí podrás ver la análitica de la empresa y satisfacción de tus consumidores.</p>'
    st.markdown(toBusiness, unsafe_allow_html=True)

    new_title = '<iframe title="Report Section" width="800" height="480" src="https://app.powerbi.com/view?r=eyJrIjoiN2M0M2Q4MGQtNjAwNi00ODFhLTk4MzMtMDU0Y2FlMjQ5ZDcxIiwidCI6ImRmODY3OWNkLWE4MGUtNDVkOC05OWFjLWM4M2VkN2ZmOTVhMCJ9" frameborder="0" allowFullScreen="true"></iframe>'
    st.markdown(new_title, unsafe_allow_html=True)
    pass

######################################### Esto lo hizo Elí ################

# Contenido principal
if sidebar_option == "Inicio":

    st.title("Bienvenido a InnovaAI Analytics!")

    #st.markdown("Explora restaurantes y descubre nuevas recomendaciones.")
    
    new_title = '<p style="font-family:Source Sans Pro; color:Black; font-size: 36px;">Explora y descubre nuevos lugares.</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/xjMxCsnc/pexels-codioful-formerly-gradienta-7130497.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.image("https://github.com/mreliflores/PF-Henry/blob/main/Sprint%233/streamlit/foto_1.jpg?raw=true", width=650)
    
elif sidebar_option == "Decido yo":
    recommend_atributo()

elif sidebar_option == "Decide el mejor":
    recommend_foodie()   

######################################### Esto lo hizo Elí ################
elif sidebar_option == "Dashboard":
    dashboard() 
######################################### Esto lo hizo Elí ################