{% set cmd_prefix = "singularity exec --nv --bind $(pwd) $PLANCKTON_SIMG " %}
{% extends base_script %}
{% block project_header %}
{{ super() }}
{% endblock %}
