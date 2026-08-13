// energybench arm `idle` -- the near-idle control.
//
// The kernel does nothing at all, so a launch of this arm costs exactly one
// round of the launch machinery: the go-message, firmware entry, the return to
// the done state. Everything the other arms measure is measured *against* this,
// which is why it has to exist as a real launched kernel rather than as "do not
// launch anything".
void kernel_main() {}
