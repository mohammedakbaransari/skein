{{/* deploy/helm/templates/_helpers.tpl */}}

{{- define "skein.name" -}}
skein
{{- end -}}

{{- define "skein.fullname" -}}
skein-agents
{{- end -}}

{{- define "skein.labels" -}}
app: {{ include "skein.name" . }}
component: agents
version: v1
{{- end -}}

{{- define "skein.selectorLabels" -}}
app: {{ include "skein.name" . }}
component: agents
{{- end -}}
