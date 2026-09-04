"""The gRPC actuation interface between a controller and the plant.

See `server.serve` to host a plant, and `client.ActuationClient` to drive one.
The schema lives in `allotrope.proto`; regenerate stubs with
`python scripts/gen_proto.py` after editing it.
"""
