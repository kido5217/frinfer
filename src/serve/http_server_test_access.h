#pragma once

#include "serve/http_server.h"

namespace ninfer::serve {

// Test-only entry to private handlers without starting the accept loop. `listen()`
// requires an attached Engine, so the engine-not-ready branches are otherwise unreachable
// from any test; production startup binding is unchanged.
struct HttpServerTestAccess {
    static void handle_metrics(const HttpServer& server, const httplib::Request& request,
                               httplib::Response& response) {
        server.handle_metrics(request, response);
    }
};

} // namespace ninfer::serve
