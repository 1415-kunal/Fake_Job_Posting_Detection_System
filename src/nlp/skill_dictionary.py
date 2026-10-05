"""
Large skill dictionary for job-posting NLP extraction.

Structure:
    SKILL_ALIASES = {
        "Canonical Skill": ["alias1", "alias2", ...],
    }

The extractor should return canonical skill names.
"""

SKILL_ALIASES = {
    # =========================
    # Programming Languages
    # =========================
    "Python": ["python", "python3", "python 3"],
    "Java": ["java"],
    "C": ["c programming", "c language"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "Go": ["golang", "go language"],
    "Rust": ["rust"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "R": ["r programming", "r language"],
    "Scala": ["scala"],
    "MATLAB": ["matlab"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "Dart": ["dart"],
    "Perl": ["perl"],
    "Lua": ["lua"],
    "Objective-C": ["objective-c", "objective c"],
    "Visual Basic": ["visual basic", "vb.net", "vb net"],
    "Assembly": ["assembly language", "asm"],
    "Groovy": ["groovy"],
    "Fortran": ["fortran"],
    "COBOL": ["cobol"],
    "SQL": ["sql"],
    "PL/SQL": ["pl/sql", "plsql"],
    "T-SQL": ["t-sql", "tsql"],

    # =========================
    # Data Science / Analytics
    # =========================
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Statistical Analysis": ["statistical analysis", "statistical modeling"],
    "Exploratory Data Analysis": ["exploratory data analysis", "eda"],
    "Data Mining": ["data mining"],
    "Data Cleaning": ["data cleaning", "data cleansing"],
    "Data Wrangling": ["data wrangling", "data munging"],
    "Feature Engineering": ["feature engineering"],
    "Feature Selection": ["feature selection"],
    "Predictive Modeling": ["predictive modeling", "predictive analytics"],
    "Descriptive Analytics": ["descriptive analytics"],
    "Prescriptive Analytics": ["prescriptive analytics"],
    "Time Series Analysis": ["time series analysis", "time series forecasting"],
    "A/B Testing": ["a/b testing", "ab testing", "split testing"],
    "Hypothesis Testing": ["hypothesis testing"],
    "Regression Analysis": ["regression analysis"],
    "Probability": ["probability"],
    "Statistics": ["statistics"],
    "Linear Algebra": ["linear algebra"],
    "Optimization": ["optimization", "mathematical optimization"],
    "Causal Inference": ["causal inference"],
    "Business Analytics": ["business analytics"],
    "Quantitative Analysis": ["quantitative analysis"],

    # =========================
    # Machine Learning
    # =========================
    "Machine Learning": ["machine learning", "ml"],
    "Supervised Learning": ["supervised learning"],
    "Unsupervised Learning": ["unsupervised learning"],
    "Semi-Supervised Learning": ["semi-supervised learning"],
    "Reinforcement Learning": ["reinforcement learning", "rl"],
    "Classification": ["classification"],
    "Regression": ["regression"],
    "Clustering": ["clustering"],
    "Dimensionality Reduction": ["dimensionality reduction"],
    "Anomaly Detection": ["anomaly detection", "outlier detection"],
    "Recommendation Systems": ["recommendation systems", "recommender systems"],
    "Ranking Models": ["ranking models", "learning to rank"],
    "Model Evaluation": ["model evaluation"],
    "Cross Validation": ["cross validation", "cross-validation"],
    "Hyperparameter Tuning": ["hyperparameter tuning", "hyperparameter optimization"],
    "Grid Search": ["grid search", "gridsearchcv"],
    "Random Search": ["random search", "randomized search"],
    "Imbalanced Learning": ["imbalanced learning"],
    "SMOTE": ["smote", "synthetic minority oversampling"],
    "Ensemble Learning": ["ensemble learning"],
    "Bagging": ["bagging"],
    "Boosting": ["boosting"],
    "Transfer Learning": ["transfer learning"],
    "Online Learning": ["online learning"],
    "Active Learning": ["active learning"],
    "Explainable AI": ["explainable ai", "xai"],
    "Model Interpretability": ["model interpretability"],
    "MLOps": ["mlops", "ml ops"],

    # =========================
    # ML Algorithms
    # =========================
    "Linear Regression": ["linear regression"],
    "Logistic Regression": ["logistic regression"],
    "Ridge Regression": ["ridge regression"],
    "Lasso Regression": ["lasso regression"],
    "Elastic Net": ["elastic net"],
    "Polynomial Regression": ["polynomial regression"],
    "Decision Trees": ["decision tree", "decision trees"],
    "Random Forest": ["random forest", "random forests"],
    "Extra Trees": ["extra trees", "extra trees classifier"],
    "Gradient Boosting": ["gradient boosting"],
    "XGBoost": ["xgboost", "xgb"],
    "LightGBM": ["lightgbm", "light gbm"],
    "CatBoost": ["catboost", "cat boost"],
    "AdaBoost": ["adaboost", "ada boost"],
    "Support Vector Machines": ["support vector machine", "support vector machines", "svm"],
    "K-Nearest Neighbors": ["k-nearest neighbors", "knn", "k nearest neighbors"],
    "Naive Bayes": ["naive bayes"],
    "K-Means": ["k-means", "kmeans", "k means"],
    "DBSCAN": ["dbscan"],
    "Hierarchical Clustering": ["hierarchical clustering"],
    "PCA": ["pca", "principal component analysis"],
    "LDA": ["lda", "linear discriminant analysis"],

    # =========================
    # Deep Learning
    # =========================
    "Deep Learning": ["deep learning"],
    "Neural Networks": ["neural network", "neural networks"],
    "Artificial Neural Networks": ["artificial neural network", "ann"],
    "Convolutional Neural Networks": ["convolutional neural network", "cnn", "cnns"],
    "Recurrent Neural Networks": ["recurrent neural network", "rnn", "rnns"],
    "LSTM": ["lstm", "long short-term memory"],
    "GRU": ["gru", "gated recurrent unit"],
    "Autoencoders": ["autoencoder", "autoencoders"],
    "GANs": ["gan", "gans", "generative adversarial network"],
    "Transformers": ["transformer", "transformers"],
    "Attention Mechanisms": ["attention mechanism", "attention mechanisms"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "JAX": ["jax"],
    "ONNX": ["onnx"],
    "CUDA": ["cuda"],
    "cuDNN": ["cudnn", "cu dnn"],
    "TensorRT": ["tensorrt", "tensor rt"],

    # =========================
    # NLP
    # =========================
    "Natural Language Processing": ["natural language processing", "nlp"],
    "Text Classification": ["text classification"],
    "Text Mining": ["text mining"],
    "Text Preprocessing": ["text preprocessing"],
    "Tokenization": ["tokenization", "tokenisation"],
    "Stemming": ["stemming"],
    "Lemmatization": ["lemmatization", "lemmatisation"],
    "Part-of-Speech Tagging": ["part of speech tagging", "pos tagging"],
    "Named Entity Recognition": ["named entity recognition", "ner"],
    "Sentiment Analysis": ["sentiment analysis"],
    "Text Summarization": ["text summarization", "text summarisation"],
    "Machine Translation": ["machine translation"],
    "Question Answering": ["question answering", "qa systems"],
    "Information Extraction": ["information extraction"],
    "Text Similarity": ["text similarity"],
    "Semantic Search": ["semantic search"],
    "TF-IDF": ["tf-idf", "tfidf"],
    "Word2Vec": ["word2vec", "word2 vec"],
    "GloVe": ["glove embeddings", "glove"],
    "FastText": ["fasttext", "fast text"],
    "spaCy": ["spacy", "spa cy"],
    "NLTK": ["nltk"],
    "Hugging Face": ["hugging face", "huggingface"],
    "Sentence Transformers": ["sentence transformers", "sentence-transformers"],
    "BERT": ["bert"],
    "RoBERTa": ["roberta"],
    "DistilBERT": ["distilbert"],
    "ALBERT": ["albert"],
    "T5": ["t5"],
    "GPT": ["gpt", "generative pre-trained transformer"],
    "XLNet": ["xlnet"],

    # =========================
    # Generative AI / LLM
    # =========================
    "AI": ["AI","artificial intelligence"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "Large Language Models": ["large language models", "large language model", "llm", "llms"],
    "Prompt Engineering": ["prompt engineering"],
    "Prompt Design": ["prompt design"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Embeddings": ["embeddings", "embedding models"],
    "Vector Databases": ["vector database", "vector databases", "vector db"],
    "FAISS": ["faiss"],
    "Chroma": ["chroma", "chromadb"],
    "Pinecone": ["pinecone"],
    "Weaviate": ["weaviate"],
    "Milvus": ["milvus"],
    "LangChain": ["langchain"],
    "LlamaIndex": ["llamaindex", "llama index"],
    "OpenAI API": ["openai api", "openai apis"],
    "Google Gemini": ["google gemini", "gemini"],
    "Anthropic Claude": ["anthropic claude", "claude"],
    "Llama": ["llama", "llama 2", "llama 3"],
    "Mistral": ["mistral", "mistral ai"],
    "Ollama": ["ollama"],
    "AI Agents": ["ai agents", "ai agent", "agentic ai"],
    "Agentic AI": ["agentic ai"],
    "Function Calling": ["function calling", "tool calling"],
    "Fine-Tuning": ["fine tuning", "fine-tuning"],
    "LoRA": ["lora", "low-rank adaptation"],
    "QLoRA": ["qlora"],
    "RLHF": ["rlhf", "reinforcement learning from human feedback"],

    # =========================
    # Python Ecosystem
    # =========================
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "SciPy": ["scipy"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    "Bokeh": ["bokeh"],
    "Statsmodels": ["statsmodels"],
    "Polars": ["polars"],
    "Dask": ["dask"],
    "PySpark": ["pyspark", "py spark"],
    "Requests": ["python requests", "requests library"],
    "BeautifulSoup": ["beautifulsoup", "beautiful soup", "bs4"],
    "Selenium": ["selenium"],
    "Scrapy": ["scrapy"],
    "Pydantic": ["pydantic"],
    "Jupyter": ["jupyter", "jupyter notebook", "jupyterlab"],
    "IPython": ["ipython"],
    "Poetry": ["poetry python"],
    "Pytest": ["pytest"],
    "Pip": ["pip python", "pip"],
    "Conda": ["conda", "anaconda"],

    # =========================
    # Backend / APIs
    # =========================
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Django REST Framework": ["django rest framework", "drf"],
    "REST API": ["rest api", "rest apis", "restful api", "restful apis"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "WebSockets": ["websocket", "websockets"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "NestJS": ["nestjs", "nest js"],
    "Spring Boot": ["spring boot"],
    "Spring Framework": ["spring framework"],
    "ASP.NET": ["asp.net", "aspnet"],
    ".NET": [".net", "dotnet"],
    "Laravel": ["laravel"],
    "Ruby on Rails": ["ruby on rails", "rails"],
    "Microservices": ["microservices", "microservices architecture"],
    "API Development": ["api development"],
    "API Integration": ["api integration"],

    # =========================
    # Databases
    # =========================
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Oracle Database": ["oracle database", "oracle db"],
    "Microsoft SQL Server": ["microsoft sql server", "sql server", "mssql"],
    "SQLite": ["sqlite"],
    "MariaDB": ["mariadb"],
    "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"],
    "Cassandra": ["cassandra"],
    "DynamoDB": ["dynamodb", "dynamo db"],
    "Firebase": ["firebase"],
    "Neo4j": ["neo4j"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "OpenSearch": ["opensearch"],
    "CouchDB": ["couchdb"],
    "Couchbase": ["couchbase"],
    "Snowflake": ["snowflake"],
    "Databricks": ["databricks"],
    "BigQuery": ["bigquery", "big query"],

    # =========================
    # Data Engineering / Big Data
    # =========================
    "Data Engineering": ["data engineering"],
    "ETL": ["etl", "extract transform load"],
    "ELT": ["elt", "extract load transform"],
    "Apache Spark": ["apache spark", "spark"],
    "Apache Hadoop": ["apache hadoop", "hadoop"],
    "Apache Kafka": ["apache kafka", "kafka"],
    "Apache Airflow": ["apache airflow", "airflow"],
    "Apache Flink": ["apache flink", "flink"],
    "Apache Beam": ["apache beam", "beam"],
    "Apache Hive": ["apache hive", "hive"],
    "Apache HBase": ["apache hbase", "hbase"],
    "Apache NiFi": ["apache nifi", "nifi"],
    "Apache Storm": ["apache storm", "storm"],
    "Databricks": ["databricks"],
    "dbt": ["dbt", "data build tool"],
    "Data Warehousing": ["data warehousing", "data warehouse"],
    "Data Pipelines": ["data pipelines", "data pipeline"],
    "Data Lakes": ["data lake", "data lakes"],
    "Delta Lake": ["delta lake"],
    "Lakehouse": ["data lakehouse", "lakehouse"],

    # =========================
    # Cloud
    # =========================
    "Amazon Web Services": ["aws", "amazon web services"],
    "Microsoft Azure": ["azure", "microsoft azure"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "Amazon EC2": ["ec2", "amazon ec2"],
    "Amazon S3": ["s3", "amazon s3"],
    "AWS Lambda": ["aws lambda", "lambda"],
    "Amazon RDS": ["rds", "amazon rds"],
    "Amazon Redshift": ["redshift", "amazon redshift"],
    "Amazon SageMaker": ["sagemaker", "amazon sagemaker"],
    "Azure Functions": ["azure functions"],
    "Azure Machine Learning": ["azure machine learning", "azure ml"],
    "Google Vertex AI": ["vertex ai", "google vertex ai"],
    "Google Cloud Storage": ["google cloud storage", "gcs"],
    "Google BigQuery": ["google bigquery", "bigquery"],
    "Cloud Computing": ["cloud computing"],
    "Cloud Architecture": ["cloud architecture"],
    "Serverless": ["serverless", "serverless computing"],

    # =========================
    # DevOps / Infrastructure
    # =========================
    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "Bitbucket": ["bitbucket"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Jenkins": ["jenkins"],
    "GitHub Actions": ["github actions"],
    "GitLab CI": ["gitlab ci"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "Chef": ["chef"],
    "Puppet": ["puppet"],
    "Helm": ["helm"],
    "Argo CD": ["argo cd", "argocd"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "Linux": ["linux"],
    "Ubuntu": ["ubuntu"],
    "Bash": ["bash", "bash scripting"],
    "PowerShell": ["powershell"],
    "Nginx": ["nginx"],
    "Apache HTTP Server": ["apache http server", "apache server"],
    "Infrastructure as Code": ["infrastructure as code", "iac"],

    # =========================
    # Frontend / Web
    # =========================
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angularjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Nuxt.js": ["nuxt.js", "nuxtjs"],
    "Svelte": ["svelte"],
    "Bootstrap": ["bootstrap"],
    "Tailwind CSS": ["tailwind css", "tailwindcss"],
    "jQuery": ["jquery"],
    "Redux": ["redux"],
    "Webpack": ["webpack"],
    "Vite": ["vite"],
    "Responsive Web Design": ["responsive web design"],

    # =========================
    # Mobile
    # =========================
    "Android Development": ["android development"],
    "Android SDK": ["android sdk"],
    "iOS Development": ["ios development"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],
    "Xamarin": ["xamarin"],
    "Jetpack Compose": ["jetpack compose"],
    "SwiftUI": ["swiftui", "swift ui"],

    # =========================
    # Visualization / BI
    # =========================
    "Microsoft Excel": ["microsoft excel", "excel"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Looker": ["looker"],
    "Looker Studio": ["looker studio", "google data studio"],
    "Qlik Sense": ["qlik sense"],
    "QlikView": ["qlikview"],
    "DAX": ["dax"],
    "Power Query": ["power query"],
    "Data Visualization": ["data visualization", "data visualisation"],

    # =========================
    # Testing / QA
    # =========================
    "Software Testing": ["software testing"],
    "Unit Testing": ["unit testing"],
    "Integration Testing": ["integration testing"],
    "End-to-End Testing": ["end-to-end testing", "e2e testing"],
    "Test Automation": ["test automation"],
    "Selenium": ["selenium"],
    "Cypress": ["cypress"],
    "Playwright": ["playwright"],
    "Postman": ["postman"],
    "JUnit": ["junit"],
    "TestNG": ["testng"],
    "Pytest": ["pytest"],
    "Jest": ["jest"],
    "Mocha": ["mocha"],
    "QA": ["quality assurance", "qa"],
    "API Testing": ["api testing"],

    # =========================
    # Security
    # =========================
    "Cybersecurity": ["cybersecurity", "cyber security"],
    "Network Security": ["network security"],
    "Application Security": ["application security", "appsec"],
    "Cloud Security": ["cloud security"],
    "Information Security": ["information security", "infosec"],
    "Penetration Testing": ["penetration testing", "pentesting", "pen testing"],
    "Ethical Hacking": ["ethical hacking"],
    "OWASP": ["owasp"],
    "SIEM": ["siem"],
    "SOC": ["security operations center", "soc"],
    "Vulnerability Assessment": ["vulnerability assessment"],
    "Digital Forensics": ["digital forensics"],
    "Identity and Access Management": ["identity and access management", "iam"],
    "OAuth": ["oauth", "oauth 2.0", "oauth2"],
    "JWT": ["jwt", "json web token"],

    # =========================
    # Software Engineering
    # =========================
    "Object-Oriented Programming": ["object oriented programming", "oop"],
    "Data Structures": ["data structures", "data structure"],
    "Algorithms": ["algorithms", "algorithm design"],
    "Design Patterns": ["design patterns"],
    "System Design": ["system design"],
    "Distributed Systems": ["distributed systems"],
    "Software Architecture": ["software architecture"],
    "Agile": ["agile", "agile methodology"],
    "Scrum": ["scrum"],
    "Kanban": ["kanban"],
    "SDLC": ["sdlc", "software development life cycle"],
    "Version Control": ["version control"],
    "Code Review": ["code review", "code reviews"],

    # =========================
    # Computer Vision
    # =========================
    "Computer Vision": ["computer vision", "cv"],
    "Image Processing": ["image processing"],
    "OpenCV": ["opencv", "open cv"],
    "Object Detection": ["object detection"],
    "Image Classification": ["image classification"],
    "Image Segmentation": ["image segmentation"],
    "OCR": ["ocr", "optical character recognition"],
    "YOLO": ["yolo", "yolo v5", "yolo v8", "yolov8"],
    "Detectron2": ["detectron2"],
    "MediaPipe": ["mediapipe", "media pipe"],

    # =========================
    # MLOps / ML Deployment
    # =========================
    "MLflow": ["mlflow", "ml flow"],
    "Kubeflow": ["kubeflow"],
    "Weights & Biases": ["weights & biases", "wandb"],
    "DVC": ["dvc", "data version control"],
    "Model Deployment": ["model deployment"],
    "Model Serving": ["model serving"],
    "Model Monitoring": ["model monitoring"],
    "Feature Stores": ["feature store", "feature stores"],
    "BentoML": ["bentoml"],
    "TorchServe": ["torchserve"],
    "NVIDIA Triton": ["nvidia triton", "triton inference server"],

    # =========================
    # Blockchain / Web3
    # =========================
    "Blockchain": ["blockchain"],
    "Ethereum": ["ethereum"],
    "Solidity": ["solidity"],
    "Smart Contracts": ["smart contracts", "smart contract"],
    "Web3": ["web3", "web 3"],
    "Hyperledger": ["hyperledger"],

    # =========================
    # Project / Collaboration
    # =========================
    "Jira": ["jira"],
    "Confluence": ["confluence"],
    "Trello": ["trello"],
    "Slack": ["slack"],
    "Microsoft Teams": ["microsoft teams", "ms teams"],
    "Asana": ["asana"],
    "Notion": ["notion"],

    # =========================
    # Core CS
    # =========================
    "Operating Systems": ["operating systems", "operating system", "os"],
    "Computer Networks": ["computer networks", "computer networking"],
    "Database Management": ["database management", "dbms"],
    "Computer Architecture": ["computer architecture"],
    "Distributed Computing": ["distributed computing"],
    "Compiler Design": ["compiler design"],
    "Cryptography": ["cryptography"],
    "Networking": ["networking"],
    "TCP/IP": ["tcp/ip", "tcp ip"],
    "HTTP": ["http", "https"],
    "DNS": ["dns"],
    "REST": ["rest"],
    "JSON": ["json"],
    "XML": ["xml"],

    # =========================
    # Soft / Professional Skills
    # =========================

    # =========================
    # Soft Skills / Professional Skills
    # =========================
    "Communication": [
        "communication", "communication skills", "verbal communication",
        "written communication", "oral communication", "effective communication"
    ],
    "Active Listening": ["active listening", "active listener"],
    "Interpersonal Skills": ["interpersonal skills", "interpersonal abilities"],
    "Teamwork": ["teamwork", "team work", "working in a team", "team player"],
    "Collaboration": ["collaboration", "collaborative skills", "collaborative"],
    "Leadership": ["leadership", "leadership skills", "leadership abilities"],
    "People Management": ["people management", "managing people"],
    "Mentoring": ["mentoring", "mentor", "mentorship"],
    "Coaching": ["coaching", "coaching skills"],
    "Problem Solving": ["problem solving", "problem-solving", "problem solving skills"],
    "Critical Thinking": ["critical thinking", "critical-thinking"],
    "Analytical Thinking": ["analytical thinking", "analytical skills"],
    "Decision Making": ["decision making", "decision-making", "decision making skills"],
    "Strategic Thinking": ["strategic thinking", "strategic planning"],
    "Creativity": ["creativity", "creative thinking", "creative problem solving"],
    "Innovation": ["innovation", "innovative thinking"],
    "Adaptability": ["adaptability", "adaptable", "adaptability skills"],
    "Flexibility": ["flexibility", "flexible"],
    "Time Management": ["time management", "time-management"],
    "Organization": ["organization skills", "organizational skills", "well organized"],
    "Prioritization": ["prioritization", "prioritisation", "prioritizing"],
    "Multitasking": ["multitasking", "multi-tasking"],
    "Attention to Detail": ["attention to detail", "detail oriented", "detail-oriented"],
    "Work Ethic": ["work ethic", "strong work ethic"],
    "Accountability": ["accountability", "accountable"],
    "Responsibility": ["responsibility", "responsible"],
    "Reliability": ["reliability", "reliable"],
    "Self Motivation": ["self motivation", "self-motivation", "self motivated", "self-motivated"],
    "Initiative": ["initiative", "takes initiative", "proactive"],
    "Emotional Intelligence": ["emotional intelligence", "emotional intelligence skills"],
    "Conflict Resolution": ["conflict resolution", "conflict management"],
    "Negotiation": ["negotiation", "negotiation skills"],
    "Persuasion": ["persuasion", "persuasive skills"],
    "Presentation Skills": ["presentation skills", "presentation"],
    "Public Speaking": ["public speaking", "public speaking skills"],
    "Storytelling": ["storytelling", "story telling"],
    "Relationship Building": ["relationship building", "building relationships"],
    "Customer Service": ["customer service", "customer support skills"],
    "Client Management": ["client management", "client relationship management"],
    "Stakeholder Management": ["stakeholder management", "stakeholder engagement"],
    "Networking": ["professional networking", "networking skills"],
    "Cross Functional Collaboration": [
        "cross-functional collaboration", "cross functional collaboration",
        "cross-functional teamwork"
    ],
    "Remote Collaboration": ["remote collaboration", "distributed team collaboration"],
    "Cultural Awareness": ["cultural awareness", "cultural sensitivity"],
    "Empathy": ["empathy", "empathetic"],
    "Patience": ["patience", "patient"],
    "Resilience": ["resilience", "resilient"],
    "Stress Management": ["stress management", "working under pressure"],
    "Learning Agility": ["learning agility", "quick learner", "fast learner"],
    "Continuous Learning": ["continuous learning", "continuous improvement", "lifelong learning"],
    "Curiosity": ["curiosity", "intellectual curiosity"],
    "Growth Mindset": ["growth mindset"],
    "Ownership": ["ownership", "sense of ownership"],
    "Professionalism": ["professionalism", "professional attitude"],
    "Dependability": ["dependability", "dependable"],
    "Integrity": ["integrity", "professional integrity"],
    "Conflict Management": ["conflict management", "managing conflict"],
    "Team Building": ["team building", "team-building"],
    "Facilitation": ["facilitation", "facilitation skills"],
    "Consensus Building": ["consensus building", "building consensus"],
    "Influencing": ["influencing", "influencing skills"],
    "Cooperation": ["cooperation", "cooperative"],
    "Workplace Communication": ["workplace communication"],
    "Business Acumen": ["business acumen"],
    "Commercial Awareness": ["commercial awareness"],
    "Customer Focus": ["customer focus", "customer-focused", "customer centric", "customer-centric"],
    "Results Orientation": ["results orientation", "results-oriented", "result oriented"],
    "Goal Setting": ["goal setting", "goal-setting"],
    "Performance Management": ["performance management"],
    "Planning": ["planning skills", "planning"],
    "Project Coordination": ["project coordination", "project coordinator skills"],
    "Change Management": ["change management"],
    "Risk Management": ["risk management"],
    "Decision Making Under Pressure": ["decision making under pressure"],
    "Written Skills": ["strong writing skills", "writing skills", "written skills"],
    "Verbal Skills": ["verbal skills", "verbal communication skills"],
    "English Communication": ["english communication", "english communication skills"],
    "Presentation": ["presentations", "presentation skills"],
}

# Skills that are too short/ambiguous for naive substring matching.
# These require stricter word-boundary matching in the extractor.
SHORT_OR_AMBIGUOUS_SKILLS = {
    "C",
    "C++",
    "C#",
    "R",
    "Go",
    "AI",
    "ML",
    "RL",
    "SQL",
    "RAG",
    "QA",
    "OS",
    "CV",
    "JS",
    "TS",
    "AWS",
    "GCP",
    "S3",
    "EC2",
    "K8s",
}

# Canonicalization helper:
# alias -> canonical skill name
ALIAS_TO_SKILL = {}

for canonical, aliases in SKILL_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_SKILL[alias.lower()] = canonical

# Useful metadata
TOTAL_CANONICAL_SKILLS = len(SKILL_ALIASES)
TOTAL_ALIASES = sum(len(aliases) for aliases in SKILL_ALIASES.values())
