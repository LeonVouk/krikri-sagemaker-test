"""
Handle sagemaker instance
"""
import boto3
import sagemaker


def instantiate_sagemaker_session():
    """
    start a sagemaker sessions
    """
    sess = sagemaker.Session()
    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it not exists
    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    print(f"sagemaker role arn: {role}")
    print(f"sagemaker bucket: {sess.default_bucket()}")
    print(f"sagemaker session region: {sess.boto_region_name}")
    return role, sess
