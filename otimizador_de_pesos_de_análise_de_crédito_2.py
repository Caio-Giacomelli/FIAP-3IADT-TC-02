# -*- coding: utf-8 -*-
"""Otimizador de Pesos de Análise de Crédito - Streamlit"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import streamlit as st

# Cache para carregar os dados apenas uma vez
@st.cache_data
def load_data():
    full_dt = pd.read_csv('https://raw.githubusercontent.com/Caio-Giacomelli/FIAP-3IADT-TC-02/refs/heads/main/credit_analysis_train.csv')
    columns_to_remove = ['ID', 'Age', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan', 'Occupation', 
                         'Delay_from_due_date', 'Monthly_Inhand_Salary', 'Changed_Credit_Limit', 
                         'Num_Credit_Inquiries', 'Credit_Mix', 'Credit_History_Age', 'Payment_of_Min_Amount', 
                         'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour', 
                         'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Monthly_Balance']
    dt = full_dt.drop(columns=columns_to_remove)
    dt = dt.replace('_', '', regex=True)
    dt['Num_of_Delayed_Payment'] = pd.to_numeric(dt['Num_of_Delayed_Payment'], errors='coerce')
    dt['Num_of_Loan'] = pd.to_numeric(dt['Num_of_Loan'], errors='coerce')
    dt['Annual_Income'] = pd.to_numeric(dt['Annual_Income'], errors='coerce')
    dt.drop_duplicates(inplace=True)
    dt = dt[(dt['Num_of_Delayed_Payment'] <= 30) & (dt['Num_of_Loan'] > 0) & (dt['Num_of_Loan'] <= 30)]
    dt = dt[(dt['Interest_Rate'] <= 100) & (dt['Num_Bank_Accounts'] <= 30) & (dt['Num_Credit_Card'] <= 30)]
    dt = dt[dt['Annual_Income'] <= 1_000_000]
    dt.replace({"Credit_Score": {"Poor": 0, "Standard": 1, "Good": 2}}, inplace=True)
    return dt

# Processamento de dados
@st.cache_resource
def preprocess_data(dt):
    X = dt.drop(columns='Credit_Score')
    y = dt['Credit_Score']
    num_pipe = Pipeline([('scaler', RobustScaler())])
    transformer = ColumnTransformer([('num_pipe', num_pipe, ['Num_of_Delayed_Payment', 'Num_of_Loan', 
                                                             'Interest_Rate', 'Num_Bank_Accounts', 
                                                             'Num_Credit_Card', 'Annual_Income'])])
    transformer.fit(X)
    X_transformed = pd.DataFrame(transformer.transform(X), columns=transformer.get_feature_names_out())
    return X_transformed, y

# Algoritmo Genético
def genetic_algorithm_streamlit(X_train, y_train, pop_size=10, num_generations=10):
    num_features = X_train.shape[1]
    population = initialize_population(pop_size, num_features)
    best_solution = None
    best_fitness = -np.inf
    fitness_history = []

    for generation in range(num_generations):
        fitness = np.array([fitness_function(ind, y_train) for ind in population])
        fitness_history.append(fitness.max())
        next_population = []

        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        next_population.append(population[fitness.argmax()])  # Elitismo
        population = np.array(next_population)

        if fitness.max() > best_fitness:
            best_fitness = fitness.max()
            best_solution = population[fitness.argmax()]

        st.line_chart(fitness_history)
        st.write(f"Geração {generation + 1}: Melhor aptidão = {best_fitness:.4f}")

    return best_solution

# Funções auxiliares
def fitness_function(chromosome, y_train):
    weights = chromosome[:-2]
    threshold_standard = chromosome[-2]
    threshold_good = chromosome[-1]
    scores = expit(np.dot(X_train_transformed, weights))
    predictions = [2 if score >= threshold_good else 1 if score >= threshold_standard else 0 for score in scores]
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

def mutate(chromosome, mutation_rate=0.5):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.uniform(-0.1, 0.1)
    return chromosome

# Interface Streamlit
st.title("Algoritmo Genético com Visualização no Streamlit")
pop_size = st.sidebar.slider("Tamanho da população", min_value=10, max_value=200, value=50, step=10)
num_generations = st.sidebar.slider("Número de gerações", min_value=10, max_value=500, value=100, step=10)

# Carregar dados e preprocessar
dt = load_data()
X_train_transformed, y_train = preprocess_data(dt)

if st.button("Iniciar Algoritmo Genético"):
    best_solution = genetic_algorithm_streamlit(
        X_train_transformed, y_train.values, pop_size=pop_size, num_generations=num_generations
    )
    st.write("Melhor solução encontrada:")
    st.write(f"Pesos: {best_solution[:-2]}")
    st.write(f"Limiar de decisão Standard: {best_solution[-2]:.4f}")
    st.write(f"Limiar de decisão Good: {best_solution[-1]:.4f}")
