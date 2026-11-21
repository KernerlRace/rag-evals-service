{{- define "rag.labels" -}}
app.kubernetes.io/name: rag
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "rag.dbUrl" -}}
postgresql://rag:{{ .Values.postgres.password }}@{{ .Release.Name }}-postgres:5432/rag
{{- end }}

{{/* One API Deployment. Rendered once for stable and once for canary, with a different retriever. */}}
{{- define "rag.apiDeployment" -}}
{{- $root := .root -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $root.Release.Name }}-api-{{ .variant }}
  labels:
    {{- include "rag.labels" $root | nindent 4 }}
    app.kubernetes.io/component: api
    rag/variant: {{ .variant }}
spec:
  replicas: {{ .cfg.replicas }}
  selector:
    matchLabels:
      app.kubernetes.io/instance: {{ $root.Release.Name }}
      app.kubernetes.io/component: api
      rag/variant: {{ .variant }}
  template:
    metadata:
      labels:
        {{- include "rag.labels" $root | nindent 8 }}
        app.kubernetes.io/component: api
        rag/variant: {{ .variant }}
    spec:
      # The API loads all chunks at startup, so it waits until the ingest Job has filled the table
      # instead of crash-looping while Postgres starts and the docs are embedded.
      initContainers:
        - name: wait-for-index
          image: {{ $root.Values.postgres.image }}
          env:
            - name: PGPASSWORD
              value: {{ $root.Values.postgres.password | quote }}
          command:
            - sh
            - -c
            - >-
              until [ "$(psql -h {{ $root.Release.Name }}-postgres -U rag -d rag -tAc
              'select count(*) from chunks' 2>/dev/null || echo 0)" -gt 0 ]; do sleep 5; done
      containers:
        - name: api
          image: "{{ $root.Values.image.repository }}:{{ $root.Values.image.tag }}"
          imagePullPolicy: {{ $root.Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
          env:
            - name: RAG_STORE
              value: postgres
            - name: DATABASE_URL
              value: {{ include "rag.dbUrl" $root | quote }}
            - name: RAG_RETRIEVER
              value: {{ .cfg.retriever | quote }}
            - name: RAG_VARIANT
              value: {{ .variant | quote }}
            - name: RAG_TOP_K
              value: {{ $root.Values.topK | quote }}
            - name: RAG_LLM_BACKEND
              value: {{ $root.Values.llm.backend | quote }}
            - name: OLLAMA_URL
              value: {{ $root.Values.llm.ollamaUrl | quote }}
            - name: OLLAMA_MODEL
              value: {{ $root.Values.llm.ollamaModel | quote }}
          # Startup loads two ONNX models and builds the BM25 index; no traffic until /health answers.
          startupProbe:
            httpGet: {path: /health, port: http}
            periodSeconds: 5
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /health, port: http}
            periodSeconds: 5
          resources:
            {{- toYaml $root.Values.resources | nindent 12 }}
{{- end }}
