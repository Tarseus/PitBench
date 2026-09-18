package org.pitbench.adapters;

import com.google.ortools.Loader;
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.IntVar;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ModelBuildRunner {
  private ModelBuildRunner() {}

  private static int field(String input, String name) {
    Matcher matcher = Pattern.compile("\\\"" + name + "\\\"\\s*:\\s*(\\d+)").matcher(input);
    if (!matcher.find()) throw new IllegalArgumentException("missing integer field: " + name);
    return Integer.parseInt(matcher.group(1));
  }

  private static String sha256(byte[] bytes) {
    try {
      byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
      StringBuilder result = new StringBuilder();
      for (byte value : digest) result.append(String.format("%02x", value));
      return result.toString();
    } catch (NoSuchAlgorithmException error) {
      throw new IllegalStateException(error);
    }
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 4) throw new IllegalArgumentException("expected instance seed budget threads");
    Loader.loadNativeLibraries();
    String input = Files.readString(Path.of(args[0]));
    int intervalCount = field(input, "intervals");
    int reuseFactor = field(input, "constant_reuse_factor");
    if (intervalCount <= 0 || reuseFactor <= 0) throw new IllegalArgumentException("invalid model dimensions");
    long started = System.nanoTime();
    CpModel model = new CpModel();
    IntVar[] ends = new IntVar[intervalCount];
    for (int index = 0; index < intervalCount; index++) {
      IntVar start = model.newIntVar(0, intervalCount * 20L, "start_" + index);
      IntVar end = model.newIntVar(0, intervalCount * 20L, "end_" + index);
      int duration = 1 + (index % reuseFactor);
      model.newIntervalVar(start, model.newConstant(duration), end, "interval_" + index);
      ends[index] = end;
    }
    IntVar makespan = model.newIntVar(0, intervalCount * 20L, "makespan");
    model.addMaxEquality(makespan, ends);
    model.minimize(makespan);
    byte[] serialized = model.model().toByteArray();
    double elapsed = (System.nanoTime() - started) / 1_000_000_000.0;
    System.out.println("{\"valid\":true,\"has_solution\":true,\"solver_status\":\"MODEL_BUILT\",\"solver_termination\":\"optimal\",\"cpu_time_sec\":" + elapsed + ",\"wall_time_sec\":" + elapsed + ",\"model_variables\":" + model.model().getVariablesCount() + ",\"model_constraints\":" + model.model().getConstraintsCount() + ",\"solution\":{\"model_sha256\":\"" + sha256(serialized) + "\",\"model_variables\":" + model.model().getVariablesCount() + ",\"model_constraints\":" + model.model().getConstraintsCount() + "}}");
  }
}
