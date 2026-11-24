# Notes

I load every PNG as a small RGB thumbnail and compare adjacent-frame likelihood using
color, pixel, edge, and histogram similarity. For small videos I solve the minimum
Hamiltonian path exactly; larger videos use cheapest-insertion followed by 2-opt
improvement. Scene cuts and very large motions can still be ambiguous, and I do not
currently use temporal metadata or audio (there is none).
