```mermaid
flowchart TD
    A[Setup & Installation] --> B[Dataset Creation]
    
    %% Dataset Creation Section
    B --> B1[Waste Management Dataset]
    B --> B2[Healthcare Dataset]
    B --> B3[Air Quality Dataset]
    B1 --> CSV1[(waste_management_dataset.csv)]
    B2 --> CSV2[(healthcare_access_dataset.csv)]
    B3 --> CSV3[(air_quality_dataset.csv)]

    %% EDA Section
    B --> C[Exploratory Data Analysis]
    C --> C1[Waste Management Analysis]
    C --> C2[Healthcare Analysis]
    C --> C3[Air Quality Analysis]
    C1 --> D[Data Correlations & Patterns]
    C2 --> D
    C3 --> D

    %% Model Training Section
    D --> E[Machine Learning Models]
    E --> E1[Illegal Dump Detection]
    E --> E2[Facility Placement]
    E --> E3[Healthcare Priority]
    E --> E4[Air Quality Prediction]
    E --> E5[Urban Zone Clustering]
    
    %% Model Types
    E1 --> RF[Random Forest]
    E2 --> GB[Gradient Boosting]
    E3 --> NN[Neural Network]
    E4 --> XGB[XGBoost]
    E5 --> DBSCAN[DBSCAN Clustering]

    %% Model Artifacts
    RF --> PKL1[(illegal_dump_detector.pkl)]
    GB --> PKL2[(facility_placement_optimizer.pkl)]
    NN --> H5[(healthcare_priority_model.h5)]
    XGB --> PKL3[(air_quality_predictor.pkl)]
    DBSCAN --> PKL4[(urban_zone_clusterer.pkl)]

    %% Deployment Section
    PKL1 & PKL2 & H5 & PKL3 & PKL4 --> F[Model Deployment]

    %% CityWISEPredictor Class
    F --> G1[analyze_waste_management]
    F --> G2[analyze_healthcare_needs]
    F --> G3[analyze_air_quality]
    F --> G4[comprehensive_urban_analysis]

    %% API Integration
    G1 & G2 & G3 & G4 --> H[Flask API Integration]
    H --> API[API Endpoints]
    
    %% API Routes
    API --> R1[/analyze/waste]
    API --> R2[/analyze/healthcare]
    API --> R3[/analyze/air-quality]
    API --> R4[/analyze/comprehensive]

    %% Styling
    classDef setup fill:#f9f,stroke:#333,stroke-width:2px
    classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef model fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef api fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef deployment fill:#fbf,stroke:#333,stroke-width:2px

    class A setup
    class B1,B2,B3,CSV1,CSV2,CSV3 dataset
    class E1,E2,E3,E4,E5,RF,GB,NN,XGB,DBSCAN model
    class R1,R2,R3,R4,API api
    class F,H deployment
```