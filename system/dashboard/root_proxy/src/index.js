const TARGET_ORIGIN = "https://fenlie.fuuu.fun";

function buildUpstreamUrl(requestUrl) {
  const sourceUrl = new URL(requestUrl);
  return new URL(`${sourceUrl.pathname}${sourceUrl.search}`, TARGET_ORIGIN);
}

function buildProxyHeaders(request, sourceUrl) {
  const headers = new Headers(request.headers);
  headers.set("host", new URL(TARGET_ORIGIN).host);
  headers.set("x-forwarded-host", sourceUrl.host);
  headers.set("x-forwarded-proto", sourceUrl.protocol.replace(":", ""));
  headers.set("x-fenlie-root-proxy", "fuuu.fun");
  return headers;
}

function rewriteLocation(locationHeader, sourceUrl) {
  if (!locationHeader) {
    return locationHeader;
  }
  try {
    const locationUrl = new URL(locationHeader, TARGET_ORIGIN);
    if (locationUrl.origin === TARGET_ORIGIN) {
      locationUrl.protocol = sourceUrl.protocol;
      locationUrl.host = sourceUrl.host;
    }
    return locationUrl.toString();
  } catch {
    return locationHeader;
  }
}

async function handle(request) {
  const sourceUrl = new URL(request.url);
  const upstreamUrl = buildUpstreamUrl(request.url);
  const requestInit = {
    method: request.method,
    headers: buildProxyHeaders(request, sourceUrl),
    redirect: "manual",
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    requestInit.body = request.body;
  }

  const upstreamResponse = await fetch(new Request(upstreamUrl.toString(), requestInit));
  const responseHeaders = new Headers(upstreamResponse.headers);
  const rewrittenLocation = rewriteLocation(responseHeaders.get("location"), sourceUrl);
  if (rewrittenLocation) {
    responseHeaders.set("location", rewrittenLocation);
  }
  responseHeaders.set("x-fenlie-public-entry", "root-nav-proxy");

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: responseHeaders,
  });
}

export default {
  async fetch(request) {
    return handle(request);
  },
};
