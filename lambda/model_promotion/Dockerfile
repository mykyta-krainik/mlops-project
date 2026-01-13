FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --no-cache-dir

COPY handler.py ${LAMBDA_TASK_ROOT}

CMD ["handler.handler"]
