# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import expit  # Função sigmoid eficiente
import streamlit as st

# Carregando os dados
@st.cache_data
def load_data():
    full_dt = pd.read_csv('https://raw.githubusercontent.com/Caio-Giacomelli/FIAP-3IADT-TC-02/refs/heads/main/credit_analysis_train.csv')

    # Removendo colunas desnecessárias
    columns_to_remove = [
        'ID', 'Age', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan',
        'Occupation', 'Delay_from_due_date', 'Monthly_Inhand_Salary',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 
        'Credit_History_Age', 'Payment_of_Min_Amount', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Payment_Behaviour', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Monthly_Balance'
    ]
    dt = full_dt.drop(columns=columns_to_remove)

    # Tratamento de dados
    dt = dt.replace('_', '', regex=True)
    dt['Num_of_Delayed_Payment'] = pd.to_numeric(dt['Num_of_Delayed_Payment'], errors='coerce')
    dt['Num_of_Loan'] = pd.to_numeric(dt['Num_of_Loan'], errors='coerce')
    dt['Annual_Income'] = pd.to_numeric(dt['Annual_Income'], errors='coerce')

    dt.drop_duplicates(inplace=True)
    dt = dt[(dt['Num_of_Delayed_Payment'] <= 30) &
            (dt['Num_of_Loan'] > 0) & (dt['Num_of_Loan'] <= 30) &
            (dt['Interest_Rate'] <= 100) &
            (dt['Num_Bank_Accounts'] <= 30) &
            (dt['Num_Credit_Card'] <= 30) &
            (dt['Annual_Income'] <= 1_000_000)]

    dt.replace({"Credit_Score": {"Poor": 0, "Standard": 1, "Good": 2}}, inplace=True)
    return dt

# Funções para o Algoritmo Genético
def fitness_function(chromosome, X_train, y_train):
    weights = chromosome[:-2]  # Pesos das variáveis
    threshold_standard = chromosome[-2]  # Limiar de decisão standard
    threshold_good = chromosome[-1] # Limiar de decisão good
    scores = expit(np.dot(X_train, weights))  
    predictions = []
    for score in scores:
      if score >= threshold_good:
        predictions.append(2)
      elif score >= threshold_standard:
        predictions.append(1)
      else:
        predictions.append(0)
    return f1_score(y_train, predictions, average='micro')

def initialize_population(pop_size, num_features):
    return np.random.uniform(-1, 1, (pop_size, num_features + 2))

def tournament_selection(population, fitness, k=3):
    selected = np.random.choice(len(population), k, replace=False)
    best = selected[np.argmax(fitness[selected])]
    return population[best]

def crossover(parent1, parent2, crossover_rate=0.8):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Cruzamento Aritmético
def arithmetic_crossover(parent1, parent2, crossover_rate=0.8, alpha=0.5):
    if np.random.rand() < crossover_rate:
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    return parent1, parent2

def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.uniform(-0.1, 0.1)
    return chromosome

# Perturbação Gaussiana
def gaussian_mutate(chromosome, mutation_rate=0.5, sigma=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.normal(0, sigma)  # Perturbação Gaussiana
    return chromosome

def genetic_algorithm_streamlit(X_train, y_train, crossover_function, mutation_function, pop_size=10, num_generations=10, crossover_rate=0.8, mutation_rate=0.5):
    num_features = X_train.shape[1]
    population = initialize_population(pop_size, num_features)
    best_solution = None
    best_fitness = -np.inf
    fitness_history = []

    # Criar um espaço reservado para o gráfico
    chart_placeholder = st.empty()
    aptitude_placeholder = st.empty()

    for generation in range(num_generations):
        fitness = np.array([fitness_function(ind, X_train, y_train) for ind in population])
        if (num_generations == 0): fitness_history.append(fitness)

        next_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1, child2 = crossover_function(parent1, parent2, crossover_rate)
            next_population.append(mutation_function(child1, mutation_rate))
            next_population.append(mutation_function(child2, mutation_rate))

        next_population.append(population[fitness.argmax()]) # Elitismo
        population = np.array(next_population)
        
        if fitness.max() > best_fitness:
            best_fitness = fitness.max()
            best_solution = population[fitness.argmax()]

        fitness_history.append(best_fitness)
        
        # Atualizar o gráfico no mesmo espaço
        fig, ax = plt.subplots()
        ax.plot(fitness_history, color="#ff4b4b", label="Melhor Aptidão")
        ax.set_title("Crescimento da Melhor Aptidão")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Aptidão")
        ax.legend()
        chart_placeholder.pyplot(fig)
        
        with aptitude_placeholder.container():                     
            st.write(f"Melhor aptidão da geração: {best_fitness}")
    
    
    return best_solution

# Interface Streamlit
st.title("Algoritmo Genético com Visualização no Streamlit")

# Carregar dados
data = load_data()

# Pré-processamento
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_pipe = Pipeline([("scaler", RobustScaler())])
transformer = ColumnTransformer([
    ("num_pipe", num_pipe, [
        'Num_of_Delayed_Payment', 'Num_of_Loan', 'Interest_Rate',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Annual_Income'
    ])
])

# Separando os dados em treino e teste
X = transformer.fit_transform(data.drop(columns=["Credit_Score"]))
y = data["Credit_Score"].values

# Dividindo a base em 70% para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configurações do algoritmo
pop_size = st.sidebar.slider("Tamanho da população", 10, 200, 50, 10)
num_generations = st.sidebar.slider("Número de gerações", 10, 500, 100, 10)
crossover_rate = st.sidebar.slider("Taxa de Cruzamento", 0.1, 1.0, 0.8, 0.1)
mutation_rate = st.sidebar.slider("Taxa de Mutação", 0.1, 1.0, 0.5, 0.1)

crossover_option = st.sidebar.selectbox(
    "Selecione o método de Cruzamento",
    ("Single Point Crossover", "Arithmetic Crossover"),
)

mutate_option = st.sidebar.selectbox(
    "Selecione o método de Mutação",
    ("Mutação com Perturbação Uniforme", "Mutação Gaussiana"),
)

if (crossover_option == "Single Point Crossover"): crossover_function = crossover
else: crossover_function = arithmetic_crossover

if (mutate_option == "Mutação Gaussiana"): mutate_function = gaussian_mutate
else: mutate_function = mutate

if st.button("Iniciar Algoritmo Genético"):
    best_solution = genetic_algorithm_streamlit(X_train, y_train, crossover_function, mutate_function, pop_size, num_generations, crossover_rate, mutation_rate)

    # Validando o modelo na base de teste
    weights = best_solution[:-2]  # Pesos das variáveis
    threshold_standard = best_solution[-2]  # Limiar de decisão para "Standard"
    threshold_good = best_solution[-1]  # Limiar de decisão para "Good"
    
    # Fazendo predições no conjunto de teste
    scores_test = expit(np.dot(X_test, weights))
    predictions_test = []
    
    for score in scores_test:
        if score >= threshold_good:
            predictions_test.append(2)  # Classe "Good"
        elif score >= threshold_standard:
            predictions_test.append(1)  # Classe "Standard"
        else:
            predictions_test.append(0)  # Classe "Poor"
    
    # Calculando a acurácia no conjunto de teste
    accuracy = accuracy_score(y_test, predictions_test)

    # Exibindo a acurácia final     
    st.write("Melhor solução encontrada:")
    st.write(f"Pesos: {best_solution[:-2]}")
    st.write(f"Limiar de decisão Standard: {best_solution[-2]:.4f}")
    st.write(f"Limiar de decisão Good: {best_solution[-1]:.4f}")

    st.write("Acurácia no conjunto de teste:")
    st.write(f"{accuracy:.4f}")
