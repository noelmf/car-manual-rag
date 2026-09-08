// Testing Library only unmounts by itself when the test globals are injected.
// They are not: the tests import what they use, so the teardown is wired here
// instead. Without it one test's markup is still on the page for the next.
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);
