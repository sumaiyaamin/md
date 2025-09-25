```mermaid
graph TD
    subgraph "CityWISE ML Pipeline"
        A[Setup & Installation] --> B[Dataset Creation]
        
        subgraph "Dataset Creation"
            B --> B1[Waste Management Dataset]
            B --> B2[Healthcare Dataset]
            B --> B3[Air Quality Dataset]
            B1 --> CSV1[(waste_management_dataset.csv)]
            B2 --> CSV2[(healthcare_access_dataset.csv)]
            B3 --> CSV3[(air_quality_dataset.csv)]
        end

        B --> C[Exploratory Data Analysis]
        
        subgraph "EDA Visualizations"
            C --> C1[Waste Management Analysis]
            C --> C2[Healthcare Analysis]
            C --> C3[Air Quality Analysis]
            C1 --> D[Data Correlations & Patterns]
            C2 --> D
            C3 --> D
        end

        D --> E[Machine Learning Models]

        subgraph "Model Training"
            E --> E1[Illegal Dump Detection]
            E --> E2[Facility Placement]
            E --> E3[Healthcare Priority]
            E --> E4[Air Quality Prediction]
            E --> E5[Urban Zone Clustering]
            
            E1 --> RF[Random Forest]
            E2 --> GB[Gradient Boosting]
            E3 --> NN[Neural Network]
            E4 --> XGB[XGBoost]
            E5 --> DBSCAN[DBSCAN Clustering]
        end

        subgraph "Model Artifacts"
            RF --> PKL1[(illegal_dump_detector.pkl)]
            GB --> PKL2[(facility_placement_optimizer.pkl)]
            NN --> H5[(healthcare_priority_model.h5)]
            XGB --> PKL3[(air_quality_predictor.pkl)]
            DBSCAN --> PKL4[(urban_zone_clusterer.pkl)]
        end

        PKL1 --> F[Model Deployment]
        PKL2 --> F
        H5 --> F
        PKL3 --> F
        PKL4 --> F

        subgraph "CityWISEPredictor Class"
            F --> G1[analyze_waste_management]
            F --> G2[analyze_healthcare_needs]
            F --> G3[analyze_air_quality]
            F --> G4[comprehensive_urban_analysis]
        end

        G1 --> H[Flask API Integration]
        G2 --> H
        G3 --> H
        G4 --> H

        H --> I[API Endpoints]
        
        subgraph "API Routes"
            I --> I1[/api/analyze/waste]
            I --> I2[/api/analyze/healthcare]
            I --> I3[/api/analyze/air-quality]
            I --> I4[/api/analyze/comprehensive]
        end
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bfb,stroke:#333,stroke-width:2px
    style F fill:#fbf,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px

classDef dataset fill:#e1f5fe,stroke:#01579b,stroke-width:2px
classDef model fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
classDef api fill:#fff3e0,stroke:#e65100,stroke-width:2px

class B1,B2,B3 dataset
class E1,E2,E3,E4,E5 model
class I1,I2,I3,I4 api
```