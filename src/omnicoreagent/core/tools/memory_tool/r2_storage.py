
from omnicoreagent.core.tools.memory_tool.s3_storage import S3MemoryBackend
from omnicoreagent.core.utils import logger



class R2MemoryBackend(S3MemoryBackend):
    """
    Production-grade Cloudflare R2 memory backend.
    
    R2 is S3-compatible but has some differences:
    - No regional endpoints (uses account-specific endpoints)
    - Automatic region ('auto' or 'wnam')
    - No transfer fees for egress
    - Slightly different consistency guarantees
    - Does not support all S3 features (e.g., Requester Pays)
    """

    def __init__(
        self,
        bucket_name: str,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "memories/",
        max_retries: int = 3,
        connect_timeout: int = 10,
        read_timeout: int = 30,
        max_pool_connections: int = 50,
        enable_encryption: bool = True,
        use_public_endpoint: bool = False,
    ):
        """
        Initialize R2 backend.
        
        Args:
            bucket_name: R2 bucket name
            account_id: Cloudflare account ID
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            prefix: Key prefix for all operations
            max_retries: Maximum retry attempts
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            max_pool_connections: Maximum connection pool size
            enable_encryption: Enable encryption (R2 encrypts by default)
            use_public_endpoint: Use public endpoint (for development/testing)
        """
        self.account_id = account_id
        self.use_public_endpoint = use_public_endpoint
        
        # R2 endpoint format
        # Private: https://<accountid>.r2.cloudflarestorage.com
        # Public: https://<bucketname>.<accountid>.r2.cloudflarestorage.com (if configured)
        if use_public_endpoint:
            endpoint_url = f"https://{bucket_name}.{account_id}.r2.cloudflarestorage.com"
        else:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        logger.info(f"Initializing R2 backend with endpoint: {endpoint_url}")
        
        # R2 doesn't support versioning in the same way as S3
        # So we disable it for R2
        enable_versioning = False
        
        # Initialize parent S3 backend with R2-specific settings
        super().__init__(
            bucket_name=bucket_name,
            prefix=prefix,
            region='auto',  # R2 uses 'auto' or 'wnam' for region
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            max_retries=max_retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_pool_connections=max_pool_connections,
            enable_versioning=enable_versioning,
            enable_encryption=enable_encryption,
        )

    def _ensure_versioning(self):
        """
        R2 doesn't support S3-style versioning.
        Override to skip versioning checks.
        """
        logger.info("R2 does not support S3-style versioning. Skipping versioning setup.")

    def _check_encryption(self):
        """
        R2 encrypts all objects by default with AES-256.
        Override to skip encryption checks.
        """
        logger.info("R2 encrypts all objects by default. Skipping encryption check.")

    def _get_put_object_params(self):
        """
        Override to remove ServerSideEncryption parameter.
        R2 handles encryption automatically.
        """
        # R2 encrypts by default, don't specify encryption parameter
        return {}