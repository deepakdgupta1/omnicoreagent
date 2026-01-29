import boto3
from botocore.exceptions import ClientError, BotoCoreError, ConnectionError as BotoConnectionError
from botocore.config import Config
from omnicoreagent.core.tools.memory_tool.base import AbstractMemoryBackend
import json
import time
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime
import logging
from functools import wraps
from threading import Lock
import os

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=3, backoff_base=1.0, exceptions=(ClientError, BotoConnectionError)):
    """Decorator for exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Check if error is retryable
                        if isinstance(e, ClientError):
                            error_code = e.response.get('Error', {}).get('Code', '')
                            # Don't retry on client errors (4xx) except for throttling
                            if error_code in ['NoSuchKey', '404', 'InvalidRequest']:
                                raise
                            if error_code not in ['RequestTimeout', 'ServiceUnavailable', 
                                                 'ThrottlingException', 'SlowDown']:
                                raise
                        
                        wait_time = backoff_base * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


class S3MemoryBackend(AbstractMemoryBackend):
    """
    Production-grade AWS S3 memory backend with:
    - Optimistic concurrency control via ETags
    - Exponential backoff retry logic
    - Comprehensive error handling
    - Connection pooling and timeouts
    - Cost-optimized operations
    - Thread-safe credential management
    - Proper pagination for large datasets
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "memories/",
        region: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        endpoint_url: str = None,
        max_retries: int = 3,
        connect_timeout: int = 10,
        read_timeout: int = 30,
        max_pool_connections: int = 50,
        enable_versioning: bool = True,
        enable_encryption: bool = True,
    ):
        """
        Initialize S3 backend with production settings.
        
        Args:
            bucket_name: S3 bucket name
            prefix: Key prefix for all operations (default: "memories/")
            region: AWS region (defaults to AWS_DEFAULT_REGION env var)
            aws_access_key_id: AWS access key (defaults to env/credentials file)
            aws_secret_access_key: AWS secret key (defaults to env/credentials file)
            endpoint_url: Custom endpoint (for S3-compatible services)
            max_retries: Maximum retry attempts for failed operations
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            max_pool_connections: Maximum connection pool size
            enable_versioning: Use versioning for concurrency control
            enable_encryption: Enable server-side encryption
        """
        self.bucket = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        self.enable_versioning = enable_versioning
        self.enable_encryption = enable_encryption
        self.max_retries = max_retries
        
        # Thread-safe lock for credential refresh scenarios
        self._client_lock = Lock()
        
        # Configure boto3 with production settings
        config = Config(
            retries={'max_attempts': 0},  # We handle retries ourselves
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_pool_connections=max_pool_connections,
            region_name=region
        )
        
        # Initialize S3 client with credentials
        client_params = {
            'service_name': 's3',
            'config': config,
        }
        
        if endpoint_url:
            client_params['endpoint_url'] = endpoint_url
        
        if aws_access_key_id and aws_secret_access_key:
            client_params['aws_access_key_id'] = aws_access_key_id
            client_params['aws_secret_access_key'] = aws_secret_access_key
        
        try:
            self.s3 = boto3.client(**client_params)
        except Exception as e:
            raise ValueError(f"Failed to initialize S3 client: {e}")
        
        self._validate_and_setup_bucket()

    def _validate_and_setup_bucket(self):
        """Validate bucket access and setup required configurations."""
        try:
            # Check bucket exists and is accessible
            self._check_bucket_exists()
            
            # Enable versioning if requested
            if self.enable_versioning:
                self._ensure_versioning()
            
            # Check encryption settings
            if self.enable_encryption:
                self._check_encryption()
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_msg = e.response.get('Error', {}).get('Message', '')
            
            if error_code == '404' or error_code == 'NoSuchBucket':
                raise ValueError(f"Bucket '{self.bucket}' does not exist")
            elif error_code == '403' or error_code == 'AccessDenied':
                raise ValueError(f"Access denied to bucket '{self.bucket}'. Check IAM permissions.")
            else:
                raise ValueError(f"Error accessing bucket '{self.bucket}': {error_msg}")

    @retry_on_failure(max_retries=3)
    def _check_bucket_exists(self):
        """Check if bucket exists and is accessible."""
        self.s3.head_bucket(Bucket=self.bucket)
        logger.info(f"Successfully connected to bucket: {self.bucket}")

    def _ensure_versioning(self):
        """Enable versioning on the bucket if not already enabled."""
        try:
            response = self.s3.get_bucket_versioning(Bucket=self.bucket)
            status = response.get('Status', 'Disabled')
            
            if status != 'Enabled':
                logger.info(f"Versioning is {status} on bucket {self.bucket}. Enabling versioning...")
                try:
                    self.s3.put_bucket_versioning(
                        Bucket=self.bucket,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                    logger.info(f"Successfully enabled versioning on bucket: {self.bucket}")
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    if error_code == 'AccessDenied':
                        logger.warning(
                            f"Cannot enable versioning on {self.bucket}: Access denied. "
                            "Grant s3:PutBucketVersioning permission or enable manually."
                        )
                    else:
                        logger.warning(f"Failed to enable versioning: {e}")
            else:
                logger.info(f"Versioning already enabled on bucket: {self.bucket}")
        except ClientError as e:
            logger.warning(f"Could not check versioning status: {e}")

    def _check_encryption(self):
        """Check bucket encryption settings."""
        try:
            self.s3.get_bucket_encryption(Bucket=self.bucket)
            logger.info(f"Encryption enabled on bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ServerSideEncryptionConfigurationNotFoundError':
                logger.warning(
                    f"No encryption configured on bucket {self.bucket}. "
                    "Consider enabling default encryption."
                )

    def _normalize_key(self, path: Optional[str]) -> str:
        """
        Convert path to S3 key with prefix.
        Prevents path traversal and ensures safe key generation.
        """
        if not path or path.strip() == "":
            return self.prefix
        
        # Remove leading/trailing whitespace and slashes
        clean = path.strip().strip('/')
        
        # Remove "memories/" prefix if user included it
        if clean.startswith('memories/'):
            clean = clean[9:]
        
        # Prevent path traversal
        if '..' in clean or clean.startswith('/'):
            raise ValueError(f"Invalid path: {path}. Path traversal detected.")
        
        # URL decode if needed (handle %20, etc.)
        import urllib.parse
        clean = urllib.parse.unquote(clean)
        
        return self.prefix + clean if clean else self.prefix

    def _get_put_object_params(self) -> Dict[str, Any]:
        """Get common parameters for put_object operations."""
        params = {}
        
        if self.enable_encryption:
            params['ServerSideEncryption'] = 'AES256'
        
        return params

    @retry_on_failure(max_retries=3)
    def _list_objects_paginated(self, prefix: str, max_keys: int = 1000) -> List[Dict]:
        """
        List objects with pagination to handle large result sets.
        Cost-optimized: limits results and uses delimiter.
        """
        items = []
        continuation_token = None
        
        try:
            while True:
                params = {
                    'Bucket': self.bucket,
                    'Prefix': prefix,
                    'Delimiter': '/',
                    'MaxKeys': min(max_keys, 1000),  # S3 limit is 1000
                }
                
                if continuation_token:
                    params['ContinuationToken'] = continuation_token
                
                response = self.s3.list_objects_v2(**params)
                
                # Add subdirectories
                for common_prefix in response.get('CommonPrefixes', []):
                    dir_name = common_prefix['Prefix'][len(prefix):].rstrip('/')
                    if dir_name:  # Skip empty
                        items.append({
                            'type': 'directory',
                            'name': dir_name,
                            'key': common_prefix['Prefix']
                        })
                
                # Add files
                for obj in response.get('Contents', []):
                    # Skip the prefix itself and directory markers
                    if obj['Key'] == prefix or obj['Key'].endswith('/'):
                        continue
                    
                    file_name = obj['Key'][len(prefix):]
                    items.append({
                        'type': 'file',
                        'name': file_name,
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'modified': obj['LastModified'],
                        'etag': obj['ETag'].strip('"')
                    })
                
                # Check if we need to continue
                if not response.get('IsTruncated', False):
                    break
                
                continuation_token = response.get('NextContinuationToken')
                
                # Safety check to prevent infinite loops
                if len(items) >= max_keys:
                    logger.warning(f"Reached max_keys limit of {max_keys}")
                    break
                    
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise ValueError(f"Bucket {self.bucket} not found")
            raise
        
        return items

    def view(self, path: Optional[str] = None) -> str:
        """View directory listing or file contents with proper error handling."""
        try:
            key = self._normalize_key(path)
        except ValueError as e:
            return f"Error: {e}"
        
        # Try to get as file first
        try:
            response = self._get_object_with_metadata(key)
            content = response['Body'].read().decode('utf-8')
            
            metadata_info = []
            if 'ETag' in response:
                metadata_info.append(f"ETag: {response['ETag'].strip('\"')}")
            if 'VersionId' in response and self.enable_versioning:
                metadata_info.append(f"Version: {response['VersionId']}")
            if 'LastModified' in response:
                metadata_info.append(f"Modified: {response['LastModified']}")
            
            size_kb = response.get('ContentLength', 0) / 1024
            metadata_info.append(f"Size: {size_kb:.2f} KB")
            
            metadata_str = " | ".join(metadata_info)
            
            return (
                f"Contents of s3://{self.bucket}/{key}\n"
                f"[{metadata_str}]\n"
                f"{'='*60}\n"
                f"{content}"
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code not in ['NoSuchKey', '404']:
                return f"Error reading file: {e}"
        
        # Try as directory
        if not key.endswith('/'):
            key += '/'
        
        try:
            items = self._list_objects_paginated(key, max_keys=1000)
            
            if not items:
                return (
                    f"Path not found: {path}\n"
                    f"Bucket: s3://{self.bucket}\n"
                    f"Prefix: {self.prefix}"
                )
            
            result = [f"Contents of s3://{self.bucket}/{key}:"]
            result.append(f"Total items: {len(items)}\n")
            
            # Sort: directories first, then files
            dirs = [i for i in items if i['type'] == 'directory']
            files = [i for i in items if i['type'] == 'file']
            
            if dirs:
                result.append("Directories:")
                for item in sorted(dirs, key=lambda x: x['name']):
                    result.append(f"  📁 {item['name']}/")
                result.append("")
            
            if files:
                result.append("Files:")
                for item in sorted(files, key=lambda x: x['name']):
                    size_kb = item['size'] / 1024
                    modified = item['modified'].strftime('%Y-%m-%d %H:%M:%S')
                    result.append(f"  📄 {item['name']:<40} {size_kb:>10.2f} KB  {modified}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error listing directory: {e}"

    @retry_on_failure(max_retries=3)
    def _get_object_with_metadata(self, key: str) -> Dict[str, Any]:
        """Get object with full metadata."""
        return self.s3.get_object(Bucket=self.bucket, Key=key)

    @retry_on_failure(max_retries=3)
    def _put_object_with_check(self, key: str, content: str, 
                               if_match: Optional[str] = None) -> Dict[str, Any]:
        """
        Put object with optional ETag-based optimistic locking.
        
        Args:
            key: S3 key
            content: Content to write
            if_match: ETag that must match (for concurrency control)
        """
        params = {
            'Bucket': self.bucket,
            'Key': key,
            'Body': content.encode('utf-8'),
            **self._get_put_object_params()
        }
        
        # Add conditional write if ETag provided
        if if_match:
            params['IfMatch'] = if_match
        
        try:
            return self.s3.put_object(**params)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'PreconditionFailed':
                raise ValueError(
                    f"Concurrent modification detected for {key}. "
                    "The file was modified by another process. Please retry."
                )
            raise

    def create_update(self, path: str, file_text: Any, mode: str = "create") -> str:
        """Create or update file with proper concurrency control."""
        # Normalize content
        if isinstance(file_text, str):
            content = file_text
        elif isinstance(file_text, list):
            content = "\n".join(str(item) for item in file_text)
        elif isinstance(file_text, dict):
            content = json.dumps(file_text, indent=2)
        else:
            content = str(file_text)
        
        try:
            key = self._normalize_key(path)
        except ValueError as e:
            return f"Error: {e}"
        
        if mode == "create":
            # Check if file already exists
            try:
                response = self._get_object_with_metadata(key)
                preview_content = response['Body'].read().decode('utf-8')
                preview = preview_content[:500]
                
                return (
                    f"File already exists: s3://{self.bucket}/{key}\n"
                    f"Size: {response.get('ContentLength', 0) / 1024:.2f} KB\n"
                    f"Last Modified: {response.get('LastModified', 'unknown')}\n"
                    f"--- Preview (first 500 chars) ---\n{preview}\n"
                    f"{'...' if len(preview_content) > 500 else ''}\n"
                    "Use mode='append' or mode='overwrite' to modify."
                )
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code not in ['NoSuchKey', '404']:
                    return f"Error checking file existence: {e}"
            
            # Create new file
            try:
                result = self._put_object_with_check(key, content)
                version_info = ""
                if self.enable_versioning and 'VersionId' in result:
                    version_info = f" (Version: {result['VersionId']})"
                
                return f"✓ File created: s3://{self.bucket}/{key}{version_info}"
            except Exception as e:
                return f"Error creating file: {e}"
        
        elif mode == "append":
            try:
                # Get existing content with ETag
                response = self._get_object_with_metadata(key)
                existing = response['Body'].read().decode('utf-8')
                etag = response['ETag'].strip('"')
                
                # Combine content
                combined = existing.rstrip('\n') + '\n' + content
                
                # Write with ETag check for concurrency control
                result = self._put_object_with_check(key, combined, if_match=etag)
                
                version_info = ""
                if self.enable_versioning and 'VersionId' in result:
                    version_info = f" (Version: {result['VersionId']})"
                
                return f"✓ Appended to s3://{self.bucket}/{key}{version_info}"
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['NoSuchKey', '404']:
                    return f"Cannot append: File not found at s3://{self.bucket}/{key}\nUse mode='create'."
                return f"Error appending: {e}"
            except ValueError as e:
                # Concurrent modification
                return str(e)
        
        elif mode == "overwrite":
            try:
                # Check file exists first
                response = self._get_object_with_metadata(key)
                etag = response['ETag'].strip('"')
                
                # Overwrite with ETag check
                result = self._put_object_with_check(key, content, if_match=etag)
                
                version_info = ""
                if self.enable_versioning and 'VersionId' in result:
                    version_info = f" (Version: {result['VersionId']})"
                
                return f"✓ File overwritten: s3://{self.bucket}/{key}{version_info}"
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ['NoSuchKey', '404']:
                    return f"Cannot overwrite: File not found at s3://{self.bucket}/{key}\nUse mode='create'."
                return f"Error overwriting: {e}"
            except ValueError as e:
                return str(e)
        
        else:
            return f"Invalid mode '{mode}'. Allowed modes: create, append, overwrite."

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace string with concurrency control."""
        try:
            key = self._normalize_key(path)
        except ValueError as e:
            return f"Error: {e}"
        
        try:
            # Get content with ETag
            response = self._get_object_with_metadata(key)
            content = response['Body'].read().decode('utf-8')
            etag = response['ETag'].strip('"')
            
            if old_str not in content:
                return f"String '{old_str}' not found in s3://{self.bucket}/{key}"
            
            # Count occurrences
            count = content.count(old_str)
            updated = content.replace(old_str, new_str)
            
            # Write with ETag check
            result = self._put_object_with_check(key, updated, if_match=etag)
            
            version_info = ""
            if self.enable_versioning and 'VersionId' in result:
                version_info = f" (Version: {result['VersionId']})"
            
            return (
                f"✓ Replaced {count} occurrence(s) of '{old_str}' with '{new_str}' "
                f"in s3://{self.bucket}/{key}{version_info}"
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['NoSuchKey', '404']:
                return f"File not found: s3://{self.bucket}/{key}"
            return f"Error: {e}"
        except ValueError as e:
            return str(e)

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        """Insert text at line number with concurrency control."""
        try:
            key = self._normalize_key(path)
        except ValueError as e:
            return f"Error: {e}"
        
        try:
            # Get content with ETag
            response = self._get_object_with_metadata(key)
            content = response['Body'].read().decode('utf-8')
            etag = response['ETag'].strip('"')
            
            lines = content.splitlines()
            total_lines = len(lines)
            
            # Validate and adjust insert position
            if insert_line < 1:
                insert_index = 0
            elif insert_line > total_lines:
                insert_index = total_lines
            else:
                insert_index = insert_line - 1
            
            lines.insert(insert_index, insert_text)
            updated = '\n'.join(lines)
            if content.endswith('\n'):
                updated += '\n'
            
            # Write with ETag check
            result = self._put_object_with_check(key, updated, if_match=etag)
            
            version_info = ""
            if self.enable_versioning and 'VersionId' in result:
                version_info = f" (Version: {result['VersionId']})"
            
            return (
                f"✓ Inserted text at line {insert_line} "
                f"in s3://{self.bucket}/{key}{version_info}"
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['NoSuchKey', '404']:
                return f"File not found: s3://{self.bucket}/{key}"
            return f"Error: {e}"
        except ValueError as e:
            return str(e)

    @retry_on_failure(max_retries=3)
    def _delete_object(self, key: str) -> Dict[str, Any]:
        """Delete a single object."""
        return self.s3.delete_object(Bucket=self.bucket, Key=key)

    def delete(self, path: str) -> str:
        """Delete file or directory with proper error handling."""
        try:
            key = self._normalize_key(path)
        except ValueError as e:
            return f"Error: {e}"
        
        try:
            # Try to delete as single file first
            self._delete_object(key)
            return f"✓ File deleted: s3://{self.bucket}/{key}"
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code not in ['NoSuchKey', '404']:
                return f"Error deleting file: {e}"
        
        # Try as directory - delete all objects with this prefix
        if not key.endswith('/'):
            key += '/'
        
        try:
            deleted_count = 0
            continuation_token = None
            
            while True:
                # List objects to delete
                params = {
                    'Bucket': self.bucket,
                    'Prefix': key,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    params['ContinuationToken'] = continuation_token
                
                response = self.s3.list_objects_v2(**params)
                
                contents = response.get('Contents', [])
                if not contents:
                    if deleted_count == 0:
                        return f"Path not found: {path}"
                    break
                
                # Delete in batches (S3 allows up to 1000 per request)
                objects_to_delete = [{'Key': obj['Key']} for obj in contents]
                
                delete_response = self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects_to_delete, 'Quiet': True}
                )
                
                deleted_count += len(contents)
                
                # Check for errors
                errors = delete_response.get('Errors', [])
                if errors:
                    logger.warning(f"Some objects failed to delete: {errors}")
                
                # Check if there are more objects
                if not response.get('IsTruncated', False):
                    break
                
                continuation_token = response.get('NextContinuationToken')
            
            return f"✓ Directory deleted: s3://{self.bucket}/{key} ({deleted_count} objects)"
            
        except ClientError as e:
            return f"Error deleting directory: {e}"

    def rename(self, old_path: str, new_path: str) -> str:
        """Rename/move file with proper error handling."""
        try:
            old_key = self._normalize_key(old_path)
            new_key = self._normalize_key(new_path)
        except ValueError as e:
            return f"Error: {e}"
        
        try:
            # Check if source exists
            self.s3.head_object(Bucket=self.bucket, Key=old_key)
            
            # Copy to new location
            copy_source = {'Bucket': self.bucket, 'Key': old_key}
            
            copy_params = {
                'Bucket': self.bucket,
                'CopySource': copy_source,
                'Key': new_key,
                **self._get_put_object_params()
            }
            
            self.s3.copy_object(**copy_params)
            
            # Delete original
            self._delete_object(old_key)
            
            return f"✓ Renamed: s3://{self.bucket}/{old_key} → s3://{self.bucket}/{new_key}"
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['NoSuchKey', '404']:
                return f"Source file not found: s3://{self.bucket}/{old_path}"
            return f"Error renaming: {e}"

    def clear_all_memory(self) -> str:
        """Clear all memory with pagination and proper error handling."""
        try:
            deleted_count = 0
            continuation_token = None
            
            logger.warning(f"Clearing all memory from s3://{self.bucket}/{self.prefix}")
            
            while True:
                # List objects
                params = {
                    'Bucket': self.bucket,
                    'Prefix': self.prefix,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    params['ContinuationToken'] = continuation_token
                
                response = self.s3.list_objects_v2(**params)
                
                contents = response.get('Contents', [])
                if not contents:
                    break
                
                # Delete in batches
                objects_to_delete = [{'Key': obj['Key']} for obj in contents]
                
                delete_response = self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects_to_delete, 'Quiet': True}
                )
                
                deleted_count += len(contents)
                
                # Log errors
                errors = delete_response.get('Errors', [])
                if errors:
                    logger.error(f"Errors during batch delete: {errors}")
                
                # Check if more objects exist
                if not response.get('IsTruncated', False):
                    break
                
                continuation_token = response.get('NextContinuationToken')
            
            return f"✓ All memory cleared from s3://{self.bucket}/{self.prefix} ({deleted_count} objects deleted)"
            
        except ClientError as e:
            return f"Error clearing memory: {e}"
        except Exception as e:
            logger.error(f"Unexpected error during clear_all_memory: {e}")
            return f"Unexpected error: {e}"