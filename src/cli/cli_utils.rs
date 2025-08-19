pub fn cli_args_for_export(args: Vec<String>) -> bool {
    if args.len() < 2 {
        return false;
    } else {
        let export = &args[1];

        if export == "export" {
            return true;
        } else {
            return false;
        }
    }
}
