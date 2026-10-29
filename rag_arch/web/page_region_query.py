"""Small region query contract for protected page viewers and citations."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..page_regions import normalize_region


def parse_region_query(value):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError("区域参数必须是四个逗号分隔的归一化坐标。")
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        raise ValueError("区域参数必须是四个逗号分隔的归一化坐标。") from None
    return normalize_region(values)


def region_query_value(region):
    values = normalize_region(region)
    return None if values is None else ",".join(repr(value) for value in values)


def page_source_url(base, source):
    """Use explicit page metadata; region IDs do not encode a parseable suffix."""
    page = source.get("page")
    if not base or type(page) is not int or page < 1:
        return base
    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["page"] = str(page)
    if query.get("asset"):
        # A model's plan:current alias belongs to its task-local registry. The
        # protected single-upload viewer has its own upload:<asset> registry;
        # translate only the viewer URL, retaining the model's provenance row.
        query["source_ref"] = "upload:" + query["asset"]
    elif source.get("source_ref"):
        query["source_ref"] = str(source["source_ref"])
    region = region_query_value(source.get("region"))
    if region is not None:
        query["region"] = region
    else:
        query.pop("region", None)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
