import concurrent.futures
import os
import warnings
from contextlib import contextmanager
from urllib.parse import urlparse, unquote

import blobfile as bf
from filelock import FileLock

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

class NotBlobPathException(Exception):
    pass

def parse_url(url):
    """
    Given a GCS or Azure path, returns either:
        ('gs', bucket, path)
     or ('az', account, path)
    """
    result = urlparse(url.rstrip("/"))
    if result.scheme == "gs":
        return 'gs', result.netloc, unquote(result.path.lstrip("/"))
    elif result.scheme == "https" and result.netloc == "storage.googleapis.com":
        bucket, rest = result.path.lstrip("/").split("/", 1)
        return 'gs', bucket, unquote(rest)
    elif result.scheme == "https" and result.netloc.endswith(".blob.core.windows.net"):
        account = result.netloc[:-len(".blob.core.windows.net")]
        return 'az', account, unquote(result.path.lstrip("/"))
    else:
        raise NotBlobPathException(f"Could not parse {url} as blob storage url")

def is_blob_url(url):
    try:
        parse_url(url)
        return True
    except NotBlobPathException:
        return False

## 수정 6.23
# def parallel_copy_recursive(src_dir, dst_dir, max_workers=16, overwrite=False):
#     """Similar to `gsutil -m cp -r $local_dir/'*' $remote_dir/`"""
#     futures = []
#     # NOTE: if we use ProcessPoolExecutor, this can't be used within pytorch workers
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for root, _, filenames in bf.walk(src_dir):
#             print(f"src_dir: {src_dir}, root: {root}") # 추가
#             # assert root.startswith(src_dir) # 생략
#             for filename in filenames:
#                 src_file = bf.join(root, filename)
#                 dst_file = bf.join(dst_dir, root[len(src_dir) + 1 :], filename)
#                 print("copying", src_file, dst_file)
#                 future = executor.submit(bf.copy, src_file, dst_file, overwrite=overwrite)
#                 futures.append(future)
#         for future in futures:
#             future.result()
def parallel_copy_recursive(src_dir, dst_dir, max_workers=16, overwrite=False):
    scheme, account, path = parse_url(src_dir)
    if scheme == "az":
        src_dir_blob = f"az://{account}/{path}"
    elif scheme == "gs":
        src_dir_blob = f"gs://{account}/{path}"
    else:
        src_dir_blob = src_dir  # fallback

    src_dir_blob = src_dir_blob.rstrip('/')
    print(f"src_dir_blob: {src_dir_blob}")
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, filenames in bf.walk(src_dir):
            print(f"root: {root}, files: {filenames}")
            for filename in filenames:
                src_file = bf.join(root, filename)
                if root.startswith(src_dir_blob):
                    rel_path = root[len(src_dir_blob):].lstrip('/')
                else:
                    rel_path = os.path.relpath(root, src_dir_blob)
                # rel_path가 ''이면, dst_file = dst_dir/filename
                # rel_path가 'checkpoint'면, dst_file = dst_dir/checkpoint/filename
                dst_file = os.path.join(dst_dir, *rel_path.split('/'), filename) if rel_path else os.path.join(dst_dir, filename)
                print(f"copying {src_file} -> {dst_file}")
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                future = executor.submit(bf.copy, src_file, dst_file, overwrite=overwrite)
                futures.append(future)
        for future in futures:
            future.result()

def download_directory_cached(url):
    """ Given a blob storage path, caches the contents locally.
    WARNING: only use this function if contents under the path won't change!
    """
    cache_dir = "/tmp/bf-dir-cache"
    scheme, bucket_or_account, path = parse_url(url)
    local_path = os.path.join(cache_dir, scheme, bucket_or_account, path)

    os.makedirs(local_path, exist_ok=True)
    with FileLock(os.path.join(local_path, ".LOCKFILE")):
        downloaded_marker = os.path.join(local_path, ".DOWNLOADED")
        if not os.path.exists(downloaded_marker):
            parallel_copy_recursive(url, local_path)
            with open(downloaded_marker, "w"):
                pass

    return local_path


@contextmanager
def open_file_cached(path, mode="r"):
    """ Given a GCS path url, caches the contents locally.
    WARNING: only use this function if contents under the path won't change!
    """
    with bf.BlobFile(path, mode=mode, cache_dir="/tmp/bf-file-cache", streaming=False) as f:
        yield f
