llava13b_template = "{%- for message in messages -%}\
    {%- if message['role'] == 'user' -%}\
        USER: {{ message['content'] }}\
    {%- elif message['role'] == 'assistant' -%}\
        ASSISTANT: {{ message['content'] }}\
    {%- endif -%}\
    {%- if not loop.last -%}\
        {{ ' ' }}\
    {%- endif -%}\
{%- endfor -%}"