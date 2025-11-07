import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="AWS Cost Analysis", layout="wide")
st.title("🌩️ AWS Cost Analysis Dashboard with ML")
st.markdown("Comparative EDA with Machine Learning Insights for EC2 and S3")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# LOAD DATASETS
# ============================================================================
@st.cache_data
def load_data():
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        ec2_path = os.path.join(project_root, 'aws_resources_compute.csv')
        s3_path = os.path.join(project_root, 'aws_resources_S3.csv')
        
        if not os.path.exists(ec2_path):
            ec2_path = 'aws_resources_compute.csv'
            s3_path = 'aws_resources_S3.csv'
        
        ec2_df = pd.read_csv(ec2_path)
        s3_df = pd.read_csv(s3_path)
    except FileNotFoundError as e:
        st.error(f"Error: Could not find CSV files.")
        st.stop()
    
    ec2_df = ec2_df.dropna(how='all')
    s3_df = s3_df.dropna(how='all')
    
    ec2_df['CreationDate'] = pd.to_datetime(ec2_df['CreationDate'], errors='coerce')
    s3_df['CreationDate'] = pd.to_datetime(s3_df['CreationDate'], errors='coerce')
    
    today = pd.Timestamp.now()
    
    ec2_df['HoursActive'] = ((today - ec2_df['CreationDate']).dt.total_seconds() / 3600).clip(lower=1)
    ec2_df['TotalCostToDate'] = ec2_df['CostUSD'] * ec2_df['HoursActive']
    
    s3_df['TotalCostToDate'] = s3_df['CostUSD']
    
    ec2_df['CPUUtilization'] = pd.to_numeric(ec2_df['CPUUtilization'], errors='coerce')
    ec2_df['MemoryUtilization'] = pd.to_numeric(ec2_df['MemoryUtilization'], errors='coerce')
    ec2_df['CostUSD'] = pd.to_numeric(ec2_df['CostUSD'], errors='coerce')
    
    ec2_df['CPUUtilization'].fillna(ec2_df['CPUUtilization'].median(), inplace=True)
    ec2_df['MemoryUtilization'].fillna(ec2_df['MemoryUtilization'].median(), inplace=True)
    ec2_df['CostUSD'].fillna(0, inplace=True)
    
    return ec2_df, s3_df

ec2_df, s3_df = load_data()

# ============================================================================
# ML FUNCTIONS
# ============================================================================
@st.cache_data
def predict_ec2_cost(ec2_df):
    """Predict EC2 costs based on CPU and Memory utilization"""
    X = ec2_df[['CPUUtilization', 'MemoryUtilization']].values
    y = ec2_df['TotalCostToDate'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model, X, y

@st.cache_data
def cluster_ec2_instances(ec2_df):
    """Cluster EC2 instances by cost behavior"""
    X = ec2_df[['CPUUtilization', 'MemoryUtilization', 'TotalCostToDate']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, scaler, X_scaled

@st.cache_data
def detect_anomalies_ec2(ec2_df):
    """Detect anomalous EC2 instances using Elliptic Envelope"""
    X = ec2_df[['CPUUtilization', 'MemoryUtilization', 'TotalCostToDate']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    anomalies = ee.fit_predict(X_scaled)
    
    return anomalies, scaler

@st.cache_data
def cluster_s3_buckets(s3_df):
    """Cluster S3 buckets by storage and cost patterns"""
    X = s3_df[['TotalSizeGB', 'TotalCostToDate', 'ObjectCount']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, scaler, X_scaled

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Section", [
    "Overview",
    "EC2 Analysis",
    "S3 Analysis",
    "EC2 ML Insights",
    "S3 ML Insights",
    "Anomaly Detection",
    "Cost Summary"
])

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 EC2 Dataset")
        st.metric("Total Instances", len(ec2_df))
        st.metric("Total EC2 Cost", f"${ec2_df['TotalCostToDate'].sum():,.2f}")
        st.metric("Regions", ec2_df['Region'].nunique())
        st.metric("Instance Types", ec2_df['InstanceType'].nunique())
    
    with col2:
        st.subheader("🪣 S3 Dataset")
        st.metric("Total Buckets", len(s3_df))
        st.metric("Total S3 Cost", f"${s3_df['TotalCostToDate'].sum():,.2f}")
        st.metric("Regions", s3_df['Region'].nunique())
        st.metric("Storage Classes", s3_df['StorageClass'].nunique())
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Total AWS Cost:** ${ec2_df['TotalCostToDate'].sum() + s3_df['TotalCostToDate'].sum():,.2f}")
    with col2:
        pct = (ec2_df['TotalCostToDate'].sum() / (ec2_df['TotalCostToDate'].sum() + s3_df['TotalCostToDate'].sum()) * 100)
        st.info(f"**EC2 % of Total:** {pct:.1f}%")
    with col3:
        pct = (s3_df['TotalCostToDate'].sum() / (ec2_df['TotalCostToDate'].sum() + s3_df['TotalCostToDate'].sum()) * 100)
        st.info(f"**S3 % of Total:** {pct:.1f}%")

# ============================================================================
# PAGE: EC2 ANALYSIS
# ============================================================================
elif page == "EC2 Analysis":
    st.header("EC2 Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ec2_df['CPUUtilization'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title('CPU Utilization Distribution', fontweight='bold', fontsize=12)
        ax.set_xlabel('CPU Utilization (%)')
        ax.set_ylabel('Frequency')
        ax.axvline(ec2_df['CPUUtilization'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {ec2_df['CPUUtilization'].mean():.1f}%")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(ec2_df['CPUUtilization'], ec2_df['TotalCostToDate'], alpha=0.6, s=100, 
                            c=ec2_df['TotalCostToDate'], cmap='viridis')
        ax.set_title('CPU Utilization vs Cost', fontweight='bold', fontsize=12)
        ax.set_xlabel('CPU Utilization (%)')
        ax.set_ylabel('Cost (USD)')
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)

    
    st.markdown("---")
    
    st.subheader("EC2 Metrics by Region")
    ec2_metrics = ec2_df.groupby('Region').agg({
        'TotalCostToDate': ['sum', 'mean', 'count'],
        'CPUUtilization': 'mean',
        'MemoryUtilization': 'mean'
    }).round(2)
    ec2_metrics.columns = ['Total Cost', 'Avg Cost', 'Instance Count', 'Avg CPU %', 'Avg Memory %']
    st.dataframe(ec2_metrics.sort_values('Total Cost', ascending=False))
    
    st.markdown("---")
    
    st.subheader("Average EC2 Cost per Region")
    regions = ['us-east-1', 'us-west-2', 'ap-south-1', 'eu-west-1']
    cols = st.columns(4)
    for i, region in enumerate(regions):
        avg_cost = ec2_df[ec2_df['Region'] == region]['TotalCostToDate'].mean()
        with cols[i]:
            st.metric(region, f"${avg_cost:,.2f}")

# ============================================================================
# PAGE: S3 ANALYSIS
# ============================================================================
elif page == "S3 Analysis":
    st.header("S3 Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        s3_by_region = s3_df.groupby('Region')['TotalSizeGB'].sum().sort_values(ascending=False).head(10)
        ax.bar(range(len(s3_by_region)), s3_by_region.values, color='mediumseagreen', edgecolor='black')
        ax.set_xticks(range(len(s3_by_region)))
        ax.set_xticklabels(s3_by_region.index, rotation=45, ha='right')
        ax.set_title('Total Storage by Region (Top 10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Total Size (GB)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(s3_df['TotalSizeGB'], s3_df['TotalCostToDate'], alpha=0.6, s=100, 
                            c=s3_df['ObjectCount'], cmap='plasma')
        ax.set_title('Storage vs Cost', fontweight='bold', fontsize=12)
        ax.set_xlabel('Total Size (GB)')
        ax.set_ylabel('Cost (USD)')
        plt.colorbar(scatter, ax=ax, label='Object Count')
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("S3 Metrics by Region")
    s3_metrics = s3_df.groupby('Region').agg({
        'TotalSizeGB': ['sum', 'mean'],
        'TotalCostToDate': ['sum', 'mean'],
        'ObjectCount': 'sum'
    }).round(2)
    s3_metrics.columns = ['Total Storage (GB)', 'Avg Storage (GB)', 'Total Cost', 'Avg Cost', 'Total Objects']
    st.dataframe(s3_metrics.sort_values('Total Cost', ascending=False))
    
    st.markdown("---")
    
    st.subheader("Total S3 Storage per Region")
    regions = ['us-east-1', 'us-west-2', 'ap-south-1', 'eu-west-1']
    cols = st.columns(4)
    for i, region in enumerate(regions):
        total_storage = s3_df[s3_df['Region'] == region]['TotalSizeGB'].sum()
        with cols[i]:
            st.metric(region, f"{total_storage:,.2f} GB")

# ============================================================================
# PAGE: EC2 ML INSIGHTS
# ============================================================================
elif page == "EC2 ML Insights":
    st.header("EC2 Machine Learning Insights")
    
    # Cost Prediction
    st.subheader("🤖 Cost Prediction Model (Random Forest)")
    model, X, y = predict_ec2_cost(ec2_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_input = st.slider("CPU Utilization (%)", 0.0, 100.0, 50.0)
        mem_input = st.slider("Memory Utilization (%)", 0.0, 100.0, 50.0)
        
        predicted_cost = model.predict([[cpu_input, mem_input]])[0]
        st.metric("Predicted Monthly Cost", f"${predicted_cost:,.2f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = model.feature_importances_
        features = ['CPU Utilization', 'Memory Utilization']
        ax.barh(features, feature_importance, color=['#FF6B6B', '#4ECDC4'])
        ax.set_title('Feature Importance in Cost Prediction', fontweight='bold', fontsize=12)
        ax.set_xlabel('Importance')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Instance Clustering
    st.subheader("🎯 EC2 Instance Clustering (K-Means)")
    clusters, scaler, X_scaled = cluster_ec2_instances(ec2_df)
    ec2_df['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(ec2_df['CPUUtilization'], ec2_df['MemoryUtilization'], 
                            c=clusters, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax.set_title('Instance Clusters: CPU vs Memory', fontweight='bold', fontsize=12)
        ax.set_xlabel('CPU Utilization (%)')
        ax.set_ylabel('Memory Utilization (%)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    with col2:
        st.write("**Cluster Summary:**")
        for cluster_id in range(3):
            cluster_data = ec2_df[ec2_df['Cluster'] == cluster_id]
            st.write(f"**Cluster {cluster_id}:**")
            st.write(f"  - Instances: {len(cluster_data)}")
            st.write(f"  - Avg CPU: {cluster_data['CPUUtilization'].mean():.1f}%")
            st.write(f"  - Avg Memory: {cluster_data['MemoryUtilization'].mean():.1f}%")
            st.write(f"  - Avg Cost: ${cluster_data['TotalCostToDate'].mean():,.2f}")

# ============================================================================
# PAGE: S3 ML INSIGHTS
# ============================================================================
elif page == "S3 ML Insights":
    st.header("S3 Machine Learning Insights")
    
    st.subheader("🎯 S3 Bucket Clustering (K-Means)")
    clusters, scaler, X_scaled = cluster_s3_buckets(s3_df)
    s3_df['Cluster'] = clusters
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(s3_df['TotalSizeGB'], s3_df['TotalCostToDate'], 
                            c=clusters, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        ax.set_title('Bucket Clusters: Storage vs Cost', fontweight='bold', fontsize=12)
        ax.set_xlabel('Total Size (GB)')
        ax.set_ylabel('Cost (USD)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    with col2:
        st.write("**Cluster Summary:**")
        for cluster_id in range(3):
            cluster_data = s3_df[s3_df['Cluster'] == cluster_id]
            st.write(f"**Cluster {cluster_id}:**")
            st.write(f"  - Buckets: {len(cluster_data)}")
            st.write(f"  - Avg Size: {cluster_data['TotalSizeGB'].mean():,.2f} GB")
            st.write(f"  - Avg Objects: {cluster_data['ObjectCount'].mean():,.0f}")
            st.write(f"  - Avg Cost: ${cluster_data['TotalCostToDate'].mean():,.2f}")

# ============================================================================
# PAGE: ANOMALY DETECTION
# ============================================================================
elif page == "Anomaly Detection":
    st.header("Anomaly Detection")
    
    st.subheader("🚨 EC2 Instance Anomalies (Elliptic Envelope)")
    anomalies, scaler = detect_anomalies_ec2(ec2_df)
    ec2_df['Anomaly'] = anomalies
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        normal = ec2_df[ec2_df['Anomaly'] == 1]
        anomaly_data = ec2_df[ec2_df['Anomaly'] == -1]
        
        ax.scatter(normal['CPUUtilization'], normal['MemoryUtilization'], 
                  c='green', label='Normal', alpha=0.6, s=100)
        ax.scatter(anomaly_data['CPUUtilization'], anomaly_data['MemoryUtilization'], 
                  c='red', label='Anomaly', alpha=0.8, s=150, marker='X')
        
        ax.set_title('Detected Anomalies: CPU vs Memory', fontweight='bold', fontsize=12)
        ax.set_xlabel('CPU Utilization (%)')
        ax.set_ylabel('Memory Utilization (%)')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        anomaly_count = len(ec2_df[ec2_df['Anomaly'] == -1])
        st.metric("Anomalies Detected", anomaly_count)
        
        if anomaly_count > 0:
            st.write("**Anomalous Instances:**")
            anomaly_instances = ec2_df[ec2_df['Anomaly'] == -1][
                ['ResourceId', 'InstanceType', 'CPUUtilization', 'MemoryUtilization', 'TotalCostToDate']
            ].head(10)
            st.dataframe(anomaly_instances)
    
    st.markdown("---")
    
    st.subheader("Cost Anomalies Analysis")
    st.write("Instances with unusual cost patterns relative to their utilization:")
    
    ec2_df['CostPerCPU'] = ec2_df['TotalCostToDate'] / (ec2_df['CPUUtilization'] + 1)
    high_cost_instances = ec2_df.nlargest(5, 'CostPerCPU')[
        ['ResourceId', 'InstanceType', 'CPUUtilization', 'TotalCostToDate', 'CostPerCPU']
    ]
    st.dataframe(high_cost_instances)

# ============================================================================
# PAGE: COST SUMMARY
# ============================================================================
elif page == "Cost Summary":
    st.header("Cost Summary")
    
    total_ec2_cost = ec2_df['TotalCostToDate'].sum()
    total_s3_cost = s3_df['TotalCostToDate'].sum()
    total_cost = total_ec2_cost + total_s3_cost
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total EC2 Cost", f"${total_ec2_cost:,.2f}")
    
    with col2:
        st.metric("Total S3 Cost", f"${total_s3_cost:,.2f}")
    
    with col3:
        st.metric("Total AWS Cost", f"${total_cost:,.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        sizes = [total_ec2_cost, total_s3_cost]
        labels = [f'EC2\n${total_ec2_cost:,.0f}', f'S3\n${total_s3_cost:,.0f}']
        colors = ['#FF6B6B', '#4ECDC4']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        ax.set_title('Cost Distribution: EC2 vs S3', fontweight='bold', fontsize=14)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cost Breakdown")
        st.metric("EC2 % of Total", f"{(total_ec2_cost/total_cost*100):.1f}%")
        st.metric("S3 % of Total", f"{(total_s3_cost/total_cost*100):.1f}%")
        
        st.write("\n**Average Costs:**")
        st.write(f"Average EC2 instance cost: ${ec2_df['TotalCostToDate'].mean():,.2f}")
        st.write(f"Average S3 bucket cost: ${s3_df['TotalCostToDate'].mean():,.2f}")
        st.write(f"Median EC2 instance cost: ${ec2_df['TotalCostToDate'].median():,.2f}")
        st.write(f"Median S3 bucket cost: ${s3_df['TotalCostToDate'].median():,.2f}")