# MLOps Kubernetes & Kubeflow Deployment Pipeline

## 🎓 Student Profile
* **Name:** [Your Name]
* **SAP ID:** [Your SAP ID]
* **Section:** [Your Section]
* **Semester:** [Your Semester]

---

## 📺 Project Demonstration Video
▶️ **[Click Here to Watch the Project Demonstration Video](PASTE_YOUR_GOOGLE_DRIVE_OR_YOUTUBE_LINK_HERE)**

---

## 🛠️ Project Implementation Details
This repository contains a full end-to-end MLOps lifecycle deployment validated on a local cluster environment:
1. **Infrastructure as Code:** Configured and provisioned an isolated `mlops` namespace using Terraform on Minikube.
2. **Automated ML Pipeline:** Created a modular Python workflow tracking data preprocessing, model training, and metrics logging.
3. **Model Registry:** Integrated MLflow tracking to capture model parameters (Accuracy: 1.000) and dynamically transition lifecycle states to `Production`.
4. **Scalable Production Serving:** Deployed a 3-replica production pod layout managed and load-balanced via an incoming Nginx backend layer.
