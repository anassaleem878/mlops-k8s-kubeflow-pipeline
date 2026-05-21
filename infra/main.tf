resource "kubernetes_namespace" "mlops_space" {
  metadata {
    name = var.namespace
  }
}
